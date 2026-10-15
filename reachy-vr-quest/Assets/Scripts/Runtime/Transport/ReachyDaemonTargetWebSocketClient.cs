using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Net.Sockets;
using System.Reflection;
using System.Text;
using System.Threading;
using ReachyMiniTeleop.Reachy;
using UnityEngine;
using UnityEngine.Networking;
using WebSocketSharp;

namespace ReachyMiniTeleop.Transport
{
    public sealed class ReachyDaemonTargetWebSocketClient : MonoBehaviour, IReachyMessageSender
    {
        public const int DefaultApiPort = 8000;
        public const string DefaultTargetPath = "/api/move/ws/set_target";
        public const string DefaultApiPath = "/api";
        public const string DefaultPlaySoundPath = "/media/play_sound";
        public const string DefaultStopSoundPath = "/media/stop_sound";

        [Header("Reachy Daemon")]
        public string targetWebSocketUrl = "ws://192.168.0.210:8000/api/move/ws/set_target";
        public string apiBaseUrl = "http://192.168.0.210:8000/api";
        public bool autoStart = false;
        public bool autoWakeOnStart = true;
        public bool verboseLogging = false;

        [Header("Queue")]
        public int maxQueuedMessages = 5;
        [Tooltip("Legacy sync-send cap. Worker-thread sending drains the queue to the newest payload instead.")]
        public int maxSendsPerFrame = 1;

        [Header("HTTP")]
        public int requestTimeoutSeconds = 3;

        [Header("Sound")]
        public bool soundEnabled = true;
        public string defaultSoundFile = "wake_up.wav";
        [Min(1)]
        public int soundRequestTimeoutSeconds = 1;

        public event Action<string> OnReceiveData;
        public event Action<bool> ConnectionStateChanged;
        public event Action<DaemonHttpRequest> HttpRequestStarted;
        public event Action<DaemonHttpResult> HttpRequestCompleted;

        private readonly ConcurrentQueue<string> _outgoingQueue = new ConcurrentQueue<string>();
        private readonly ConcurrentQueue<string> _incomingQueue = new ConcurrentQueue<string>();
        private readonly ConcurrentQueue<MainThreadLog> _mainThreadLogs = new ConcurrentQueue<MainThreadLog>();
        private readonly ConcurrentQueue<bool> _connectionStateQueue = new ConcurrentQueue<bool>();
        private readonly ConcurrentQueue<DaemonHttpRequest> _httpRequestQueue = new ConcurrentQueue<DaemonHttpRequest>();
        private readonly ConcurrentQueue<DaemonHttpResult> _httpResultQueue = new ConcurrentQueue<DaemonHttpResult>();
        private readonly object _httpWorkLock = new object();
        private readonly ManualResetEventSlim _stopSignal = new ManualResetEventSlim(false);
        private readonly AutoResetEvent _sendSignal = new AutoResetEvent(false);
        private readonly AutoResetEvent _httpWorkSignal = new AutoResetEvent(false);
        private Coroutine _wakeCoroutine;
        private Thread _workerThread;
        private Thread _httpWorkerThread;
        private volatile bool _running;
        private volatile bool _isConnecting;
        private volatile bool _isAlive;
        private volatile bool _httpWorkerRunning;
        private bool _httpWorkInFlight;
        private bool _hasPendingHttpWork;
        private DaemonHttpWorkItem _pendingHttpWork;
        private int _workerGeneration;
        private int _maskSeed = 1;
        private AndroidJavaObject _wifiLock;
        private volatile Socket _poseSocket;
        private readonly object _poseSendLock = new object();

#if UNITY_EDITOR
        internal Action<string, int> workerLoopOverride;
        internal int sendSignalSetCountForTests;
        internal Func<Uri, string, int, long> rawHttpResponseCodeOverrideForTests;
        internal int httpWorkerStartCountForTests;
        internal int rawHttpPostStartCountForTests;
        internal int rawHttpPostMaxConcurrentForTests;
        private int _rawHttpPostActiveCountForTests;
#endif

        public bool IsRunning => _isAlive;
        public bool IsConnectedOrConnecting => _isConnecting || _isAlive;
        public int PendingSendCount => _outgoingQueue.Count;
        public bool IsSoundHttpRequestInFlight
        {
            get
            {
                lock (_httpWorkLock)
                    return _httpWorkInFlight || _hasPendingHttpWork;
            }
        }

        private void Start()
        {
            if (autoStart)
                StartClient();
        }

        private void Update()
        {
            while (_incomingQueue.TryDequeue(out var message))
            {
                if (!string.IsNullOrEmpty(message))
                    OnReceiveData?.Invoke(message);
            }

            while (_mainThreadLogs.TryDequeue(out var log))
            {
                if (log.isWarning)
                    Debug.LogWarning(log.message);
                else
                    Debug.Log(log.message);
            }

            while (_connectionStateQueue.TryDequeue(out bool isConnectedOrConnecting))
            {
                ConnectionStateChanged?.Invoke(isConnectedOrConnecting);
            }

            while (_httpRequestQueue.TryDequeue(out var request))
            {
                HttpRequestStarted?.Invoke(request);
            }

            while (_httpResultQueue.TryDequeue(out var result))
            {
                HttpRequestCompleted?.Invoke(result);
            }
        }

        private void OnDestroy()
        {
            StopClient();
            StopHttpWorker();
        }

        public void SendMessageToServer(string data)
        {
            if (string.IsNullOrWhiteSpace(data))
                return;

            // Write on the publisher thread. The worker's timed wait was coalesced to
            // about 50 ms on Android, so a full queue still left the PC at ~20 Hz.
            Socket socket = _poseSocket;
            if (socket != null)
            {
                lock (_poseSendLock)
                {
                    try
                    {
                        SendMaskedText(socket, data);
                        return;
                    }
                    catch (Exception ex)
                    {
                        EnqueueMainThreadLog($"[ReachyDaemonTargetWebSocketClient] Send error: {ex.Message}", true);
                    }
                }
            }

            int limit = Mathf.Max(1, maxQueuedMessages);
            while (_outgoingQueue.Count >= limit && _outgoingQueue.TryDequeue(out _))
            {
            }

            _outgoingQueue.Enqueue(data);
            SignalWorkerForSend();
        }

        public bool TrySetEndpoint(string value, bool restartIfRunning)
        {
            if (!IsValidTargetWebSocketUrl(value, out string normalizedUrl))
                return false;

            bool wasRunning = IsConnectedOrConnecting;
            if (wasRunning)
            {
                if (!restartIfRunning)
                    return false;

                StopClient();
            }

            targetWebSocketUrl = normalizedUrl;

            if (wasRunning)
                StartClient();

            return true;
        }

        public bool TrySetApiBaseUrl(string value)
        {
            if (!IsValidApiBaseUrl(value, out string normalizedUrl))
                return false;

            apiBaseUrl = normalizedUrl;
            return true;
        }

        public void PlayDefaultSound()
        {
            PlaySound(defaultSoundFile);
        }

        public void PlaySound(string file)
        {
            PlaySound(file, "Face sound");
        }

        public void PlaySound(string file, string operationName)
        {
            if (!soundEnabled)
            {
                ReportHttpResult(operationName, 0, "Sound disabled", string.Empty, false);
                return;
            }

            if (!TryBuildApiEndpointUrl(apiBaseUrl, DefaultPlaySoundPath, out string url))
            {
                Debug.LogError($"[ReachyDaemonTargetWebSocketClient] Invalid daemon API base URL: {apiBaseUrl}");
                return;
            }

            if (!TryBuildPlaySoundJson(file, out string json))
            {
                Debug.LogError("[ReachyDaemonTargetWebSocketClient] Enter a sound filename before playing sound.");
                return;
            }

            PostJson(url, json, operationName);
        }

        public void StopSound()
        {
            if (!TryBuildApiEndpointUrl(apiBaseUrl, DefaultStopSoundPath, out string url))
            {
                Debug.LogError($"[ReachyDaemonTargetWebSocketClient] Invalid daemon API base URL: {apiBaseUrl}");
                return;
            }

            PostJson(url, string.Empty, "Stop sound");
        }

#if UNITY_EDITOR
        internal bool TryGetQueuedSoundHttpWorkForTests(out string operationName, out string url, out string body)
        {
            lock (_httpWorkLock)
            {
                operationName = _pendingHttpWork.OperationName;
                url = _pendingHttpWork.Url;
                body = _pendingHttpWork.Body;
                return _hasPendingHttpWork;
            }
        }
#endif

        public void StartClient()
        {
            if (IsConnectedOrConnecting)
                return;

            if (!IsValidTargetWebSocketUrl(targetWebSocketUrl, out string normalizedTargetUrl))
            {
                Debug.LogError($"[ReachyDaemonTargetWebSocketClient] Invalid target WebSocket URL: {targetWebSocketUrl}");
                return;
            }

            targetWebSocketUrl = normalizedTargetUrl;
            AcquireLowLatencyWifi();
            _stopSignal.Reset();
            _running = true;
            _isConnecting = true;
            _isAlive = false;
            int generation = Interlocked.Increment(ref _workerGeneration);
            ConnectionStateChanged?.Invoke(true);

            _workerThread = new Thread(() => WorkerLoop(targetWebSocketUrl, generation))
            {
                IsBackground = true,
                Name = "ReachyDaemonTargetWebSocketThread",
                Priority = System.Threading.ThreadPriority.AboveNormal
            };
            _workerThread.Start();

            if (autoWakeOnStart)
            {
                if (_wakeCoroutine != null)
                    StopCoroutine(_wakeCoroutine);
                _wakeCoroutine = StartCoroutine(WakeIfNeededRoutine());
            }
        }

        public void StopClient()
        {
            if (_wakeCoroutine != null)
            {
                StopCoroutine(_wakeCoroutine);
                _wakeCoroutine = null;
            }

            ReleaseLowLatencyWifi();

            if (!_running && !_isConnecting && !_isAlive)
                return;

            _running = false;
            _isConnecting = false;
            _isAlive = false;
            Interlocked.Increment(ref _workerGeneration);
            _stopSignal.Set();
            _sendSignal.Set();

            try
            {
                if (_workerThread != null && _workerThread.IsAlive && _workerThread != Thread.CurrentThread)
                    _workerThread.Join(2000);
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[ReachyDaemonTargetWebSocketClient] Worker stop error: {ex.Message}");
            }
            finally
            {
                if (_workerThread != null && !_workerThread.IsAlive)
                    _workerThread = null;
                ConnectionStateChanged?.Invoke(false);
            }
        }

        private void WorkerLoop(string url, int generation)
        {
#if UNITY_EDITOR
            if (workerLoopOverride != null)
            {
                try
                {
                    workerLoopOverride(url, generation);
                }
                catch (Exception ex)
                {
                    if (IsCurrentWorker(generation))
                        EnqueueMainThreadLog($"[ReachyDaemonTargetWebSocketClient] Worker error: {ex.Message}", true);
                }
                finally
                {
                    FinishWorker(generation);
                }

                return;
            }
#endif

            WebSocket webSocket = null;

            try
            {
                webSocket = CreateWebSocket(url, generation);
                webSocket.Connect();

                while (_running && !_stopSignal.IsSet && webSocket.IsAlive)
                {
                    if (_poseSocket != null)
                    {
                        _stopSignal.Wait(200);
                        continue;
                    }

                    if (!TryDequeueLatestOutgoing(out string data))
                    {
                        _sendSignal.WaitOne(20);
                        continue;
                    }

                    try
                    {
                        webSocket.Send(data);
                    }
                    catch (Exception ex)
                    {
                        EnqueueMainThreadLog($"[ReachyDaemonTargetWebSocketClient] Send error: {ex.Message}", true);
                        break;
                    }
                }
            }
            catch (Exception ex)
            {
                if (IsCurrentWorker(generation) && _running)
                    EnqueueMainThreadLog($"[ReachyDaemonTargetWebSocketClient] Connection exception: {ex.Message}", true);
            }
            finally
            {
                try
                {
                    if (webSocket != null && webSocket.IsAlive)
                        webSocket.Close();
                }
                catch (Exception ex)
                {
                    if (IsCurrentWorker(generation))
                        EnqueueMainThreadLog($"[ReachyDaemonTargetWebSocketClient] Close error: {ex.Message}", true);
                }

                FinishWorker(generation);
            }
        }

        private void AcquireLowLatencyWifi()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (_wifiLock != null)
                return;

            try
            {
                using (var player = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
                using (var activity = player.GetStatic<AndroidJavaObject>("currentActivity"))
                using (var wifi = activity.Call<AndroidJavaObject>("getSystemService", "wifi"))
                {
                    const int wifiModeFullLowLatency = 4;
                    const int wifiModeFullHighPerf = 3;
                    try
                    {
                        _wifiLock = wifi.Call<AndroidJavaObject>(
                            "createWifiLock", wifiModeFullLowLatency, "ReachyTeleopLowLatency");
                    }
                    catch (AndroidJavaException)
                    {
                        _wifiLock = wifi.Call<AndroidJavaObject>(
                            "createWifiLock", wifiModeFullHighPerf, "ReachyTeleopLowLatency");
                    }

                    _wifiLock.Call("setReferenceCounted", false);
                    _wifiLock.Call("acquire");
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[ReachyDaemonTargetWebSocketClient] Wi-Fi lock failed: {ex.Message}");
            }
#endif
        }

        private void ReleaseLowLatencyWifi()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (_wifiLock == null)
                return;

            try
            {
                if (_wifiLock.Call<bool>("isHeld"))
                    _wifiLock.Call("release");
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[ReachyDaemonTargetWebSocketClient] Wi-Fi unlock failed: {ex.Message}");
            }
            finally
            {
                _wifiLock.Dispose();
                _wifiLock = null;
            }
#endif
        }

        private int EnableNoDelay(WebSocket webSocket)
        {
            _poseSocket = null;
            var found = new List<Socket>();
            int tuned = TuneSockets(webSocket, 0, new HashSet<object>(), found);
            if (found.Count > 0)
                _poseSocket = found[found.Count - 1];
            return tuned;
        }

        private static int TuneSockets(object obj, int depth, HashSet<object> seen, List<Socket> found)
        {
            if (obj == null || depth > 4 || !seen.Add(obj))
                return 0;

            if (obj is Socket socket)
            {
                socket.NoDelay = true;
                try
                {
                    socket.SendTimeout = 8;
                    socket.SendBufferSize = 256 * 1024;
                    socket.ReceiveBufferSize = 256 * 1024;
                }
                catch (SocketException)
                {
                }

                found.Add(socket);
                return 1;
            }

            if (obj is TcpClient tcp)
            {
                tcp.NoDelay = true;
                try
                {
                    tcp.SendBufferSize = 256 * 1024;
                    tcp.ReceiveBufferSize = 256 * 1024;
                }
                catch (SocketException)
                {
                }

                return 1 + TuneSockets(tcp.Client, depth + 1, seen, found);
            }

            Type type = obj.GetType();
            if (type.IsPrimitive || obj is string)
                return 0;

            const BindingFlags flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
            int tuned = 0;
            for (Type current = type; current != null && current != typeof(object); current = current.BaseType)
            {
                foreach (FieldInfo field in current.GetFields(flags))
                {
                    if (field.FieldType.IsPrimitive || field.FieldType == typeof(string))
                        continue;
                    if (depth > 0 && !ShouldWalk(field))
                        continue;

                    object value;
                    try
                    {
                        value = field.GetValue(obj);
                    }
                    catch (Exception)
                    {
                        continue;
                    }

                    tuned += TuneSockets(value, depth + 1, seen, found);
                }
            }

            return tuned;
        }

        private void SendMaskedText(Socket socket, string text)
        {
            byte[] payload = Encoding.UTF8.GetBytes(text);
            int seed = _maskSeed++;
            var mask = new[]
            {
                (byte)seed,
                (byte)(seed >> 8),
                (byte)(seed >> 16),
                (byte)(seed >> 24)
            };
            int maskAt;
            byte[] frame;
            if (payload.Length < 126)
            {
                frame = new byte[6 + payload.Length];
                frame[1] = (byte)(0x80 | payload.Length);
                maskAt = 2;
            }
            else
            {
                frame = new byte[8 + payload.Length];
                frame[1] = 0xFE;
                frame[2] = (byte)(payload.Length >> 8);
                frame[3] = (byte)payload.Length;
                maskAt = 4;
            }

            frame[0] = 0x81;
            Buffer.BlockCopy(mask, 0, frame, maskAt, 4);
            for (int i = 0; i < payload.Length; i++)
                frame[maskAt + 4 + i] = (byte)(payload[i] ^ mask[i & 3]);

            int sent = 0;
            while (sent < frame.Length)
            {
                int count = socket.Send(frame, sent, frame.Length - sent, SocketFlags.None);
                if (count <= 0)
                    throw new IOException("pose socket closed");
                sent += count;
            }
        }

        private static bool ShouldWalk(FieldInfo field)
        {
            if (typeof(Stream).IsAssignableFrom(field.FieldType) ||
                typeof(Socket).IsAssignableFrom(field.FieldType) ||
                typeof(TcpClient).IsAssignableFrom(field.FieldType))
            {
                return true;
            }

            string name = field.Name;
            return name.IndexOf("stream", StringComparison.OrdinalIgnoreCase) >= 0 ||
                   name.IndexOf("socket", StringComparison.OrdinalIgnoreCase) >= 0 ||
                   name.IndexOf("tcp", StringComparison.OrdinalIgnoreCase) >= 0 ||
                   name.IndexOf("client", StringComparison.OrdinalIgnoreCase) >= 0;
        }

        private WebSocket CreateWebSocket(string url, int generation)
        {
            var webSocket = new WebSocket(url)
            {
                EmitOnPing = true,
                WaitTime = TimeSpan.FromSeconds(Math.Max(1, requestTimeoutSeconds))
            };

            webSocket.OnOpen += (_, _) =>
            {
                if (!IsCurrentWorker(generation))
                    return;

                _isConnecting = false;
                _isAlive = true;
                bool tuned = EnableNoDelay(webSocket) > 0;
                EnqueueMainThreadLog(
                    tuned
                        ? "[ReachyDaemonTargetWebSocketClient] TCP no-delay enabled"
                        : "[ReachyDaemonTargetWebSocketClient] TCP no-delay socket was not found",
                    !tuned);

                if (verboseLogging)
                    EnqueueMainThreadLog($"[ReachyDaemonTargetWebSocketClient] Connected to {url}", false);
            };
            webSocket.OnMessage += (_, args) =>
            {
                string data = args.Data;
                if (!string.IsNullOrEmpty(data))
                    _incomingQueue.Enqueue(data);
            };
            webSocket.OnError += (_, args) =>
            {
                if (!IsCurrentWorker(generation))
                    return;

                _isConnecting = false;
                _isAlive = false;
                EnqueueMainThreadLog($"[ReachyDaemonTargetWebSocketClient] WebSocket error: {args.Message}", true);
                EnqueueConnectionState(false);
            };
            webSocket.OnClose += (_, args) =>
            {
                if (!IsCurrentWorker(generation))
                    return;

                _isConnecting = false;
                _isAlive = false;

                if (verboseLogging)
                    EnqueueMainThreadLog($"[ReachyDaemonTargetWebSocketClient] Closed: {args.Reason}", false);

                EnqueueConnectionState(false);
            };

            return webSocket;
        }

        private bool TryDequeueLatestOutgoing(out string latest)
        {
            latest = null;

            while (_outgoingQueue.TryDequeue(out var data))
                latest = data;

            return latest != null;
        }

        private void FinishWorker(int generation)
        {
            if (!IsCurrentWorker(generation))
                return;

            _running = false;
            _isConnecting = false;
            _isAlive = false;
            EnqueueConnectionState(false);
        }

        private bool IsCurrentWorker(int generation)
        {
            return generation == Volatile.Read(ref _workerGeneration);
        }

        private void SignalWorkerForSend()
        {
#if UNITY_EDITOR
            Interlocked.Increment(ref sendSignalSetCountForTests);
#endif
            _sendSignal.Set();
        }

        private void EnqueueConnectionState(bool isConnectedOrConnecting)
        {
            _connectionStateQueue.Enqueue(isConnectedOrConnecting);
        }

        private void EnqueueMainThreadLog(string message, bool isWarning)
        {
            _mainThreadLogs.Enqueue(new MainThreadLog(message, isWarning));
        }

        public static bool IsValidTargetWebSocketUrl(string url, out string normalizedUrl)
        {
            normalizedUrl = null;
            if (string.IsNullOrWhiteSpace(url))
                return false;

            if (!Uri.TryCreate(url.Trim(), UriKind.Absolute, out var uri))
                return false;

            if (!string.Equals(uri.Scheme, "ws", StringComparison.OrdinalIgnoreCase) &&
                !string.Equals(uri.Scheme, "wss", StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            if (string.IsNullOrWhiteSpace(uri.Host) || uri.Port <= 0 || uri.Port > 65535)
                return false;

            normalizedUrl = uri.GetComponents(UriComponents.SchemeAndServer, UriFormat.UriEscaped) + uri.AbsolutePath;
            return normalizedUrl.EndsWith(DefaultTargetPath, StringComparison.OrdinalIgnoreCase);
        }

        public static bool IsValidApiBaseUrl(string url, out string normalizedUrl)
        {
            normalizedUrl = null;
            if (string.IsNullOrWhiteSpace(url))
                return false;

            if (!Uri.TryCreate(url.Trim(), UriKind.Absolute, out var uri))
                return false;

            if (!string.Equals(uri.Scheme, "http", StringComparison.OrdinalIgnoreCase) &&
                !string.Equals(uri.Scheme, "https", StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            if (string.IsNullOrWhiteSpace(uri.Host) || uri.Port <= 0 || uri.Port > 65535)
                return false;

            normalizedUrl = (uri.GetComponents(UriComponents.SchemeAndServer, UriFormat.UriEscaped) + uri.AbsolutePath).TrimEnd('/');
            return normalizedUrl.EndsWith(DefaultApiPath, StringComparison.OrdinalIgnoreCase);
        }

        public static bool TryBuildApiEndpointUrl(string apiBaseUrl, string endpointPath, out string endpointUrl)
        {
            endpointUrl = null;

            if (!IsValidApiBaseUrl(apiBaseUrl, out string normalizedBaseUrl) ||
                string.IsNullOrWhiteSpace(endpointPath))
            {
                return false;
            }

            string normalizedPath = endpointPath.Trim();
            if (!normalizedPath.StartsWith("/", StringComparison.Ordinal))
                normalizedPath = "/" + normalizedPath;

            endpointUrl = normalizedBaseUrl + normalizedPath;
            return Uri.TryCreate(endpointUrl, UriKind.Absolute, out _);
        }

        public static bool TryBuildPlaySoundJson(string file, out string json)
        {
            json = null;
            if (string.IsNullOrWhiteSpace(file))
                return false;

            json = JsonUtility.ToJson(new PlaySoundRequest(file.Trim()));
            return true;
        }

        public static string FormatHttpResultStatus(
            string operationName,
            long responseCode,
            string error,
            string detail,
            bool success)
        {
            string operation = string.IsNullOrWhiteSpace(operationName) ? "HTTP" : operationName.Trim();
            string code = responseCode > 0 ? responseCode.ToString() : "network";
            if (success)
                return $"{operation} HTTP {code}: ok";

            string message = ExtractHttpDetail(detail);
            if (string.IsNullOrWhiteSpace(message))
                message = string.IsNullOrWhiteSpace(error) ? "failed" : error.Trim();

            return $"{operation} HTTP {code}: {TrimHttpMessage(message)}";
        }

        public static string FormatHttpRequestStatus(string operationName, string url)
        {
            string operation = string.IsNullOrWhiteSpace(operationName) ? "HTTP" : operationName.Trim();
            string target = string.IsNullOrWhiteSpace(url) ? "unknown URL" : url.Trim();
            return $"{operation} request: {target}";
        }

        private static string ExtractHttpDetail(string detail)
        {
            if (string.IsNullOrWhiteSpace(detail))
                return string.Empty;

            string trimmed = detail.Trim();
            const string detailToken = "\"detail\":\"";
            int detailStart = trimmed.IndexOf(detailToken, StringComparison.OrdinalIgnoreCase);
            if (detailStart >= 0)
            {
                int valueStart = detailStart + detailToken.Length;
                int valueEnd = trimmed.IndexOf('"', valueStart);
                if (valueEnd > valueStart)
                    return trimmed.Substring(valueStart, valueEnd - valueStart);
            }

            return trimmed;
        }

        private static string TrimHttpMessage(string message)
        {
            string trimmed = message.Trim();
            const int maxLength = 96;
            if (trimmed.Length <= maxLength)
                return trimmed;

            return trimmed.Substring(0, maxLength - 3) + "...";
        }

        private IEnumerator WakeIfNeededRoutine()
        {
            if (!IsValidApiBaseUrl(apiBaseUrl, out string baseUrl))
            {
                _wakeCoroutine = null;
                yield break;
            }

            using (UnityWebRequest stateRequest = UnityWebRequest.Get($"{baseUrl}/state/full"))
            {
                stateRequest.timeout = Mathf.Max(1, requestTimeoutSeconds);
                yield return stateRequest.SendWebRequest();

                if (stateRequest.result != UnityWebRequest.Result.Success)
                {
                    if (verboseLogging)
                        Debug.LogWarning($"[ReachyDaemonTargetWebSocketClient] State check failed: {stateRequest.error}");
                    _wakeCoroutine = null;
                    yield break;
                }

                string stateJson = stateRequest.downloadHandler != null ? stateRequest.downloadHandler.text : string.Empty;
                if (stateJson.IndexOf("\"control_mode\":\"disabled\"", StringComparison.OrdinalIgnoreCase) < 0)
                {
                    _wakeCoroutine = null;
                    yield break;
                }
            }

            yield return PostEmpty($"{baseUrl}/motors/set_mode/enabled");
            yield return PostEmpty($"{baseUrl}/move/play/wake_up", "wake up");
            _wakeCoroutine = null;
        }

        private IEnumerator PostEmpty(string url, string operationName = "POST")
        {
            ReportHttpRequest(operationName, url, null);

            using (var request = new UnityWebRequest(url, UnityWebRequest.kHttpVerbPOST))
            {
                request.downloadHandler = new DownloadHandlerBuffer();
                request.uploadHandler = new UploadHandlerRaw(Array.Empty<byte>());
                request.timeout = Mathf.Max(1, requestTimeoutSeconds);
                yield return request.SendWebRequest();

                bool success = request.result == UnityWebRequest.Result.Success;
                string detail = request.downloadHandler != null ? request.downloadHandler.text : string.Empty;
                ReportHttpResult(operationName, request.responseCode, request.error, detail, success);

                if (!success && verboseLogging)
                    Debug.LogWarning($"[ReachyDaemonTargetWebSocketClient] {FormatHttpResultStatus(operationName, request.responseCode, request.error, detail, false)}");
            }
        }

        private void PostJson(string url, string json, string operationName)
        {
            if (!Uri.TryCreate(url, UriKind.Absolute, out var uri) ||
                !string.Equals(uri.Scheme, "http", StringComparison.OrdinalIgnoreCase))
            {
                StartCoroutine(PostJsonWithUnityWebRequest(url, json, operationName));
                return;
            }

            EnqueueRawHttpWork(new DaemonHttpWorkItem(operationName, url, uri, json));
        }

        private IEnumerator PostJsonWithUnityWebRequest(string url, string json, string operationName)
        {
            ReportHttpRequest(operationName, url, json);

            using (var request = new UnityWebRequest(url, UnityWebRequest.kHttpVerbPOST))
            {
                byte[] body = Encoding.UTF8.GetBytes(json ?? string.Empty);
                request.downloadHandler = new DownloadHandlerBuffer();
                request.uploadHandler = new UploadHandlerRaw(body);
                if (!string.IsNullOrEmpty(json))
                    request.SetRequestHeader("Content-Type", "application/json");
                request.timeout = Mathf.Max(1, requestTimeoutSeconds);
                yield return request.SendWebRequest();

                bool success = request.result == UnityWebRequest.Result.Success;
                string detail = request.downloadHandler != null ? request.downloadHandler.text : string.Empty;
                ReportHttpResult(operationName, request.responseCode, request.error, detail, success);

                if (!success)
                {
                    Debug.LogWarning($"[ReachyDaemonTargetWebSocketClient] {FormatHttpResultStatus(operationName, request.responseCode, request.error, detail, false)}");
                }
                else if (verboseLogging)
                {
                    Debug.Log($"[ReachyDaemonTargetWebSocketClient] {FormatHttpResultStatus(operationName, request.responseCode, request.error, detail, true)} sent to {url}");
                }
            }
        }

        private void ReportHttpRequest(string operationName, string url, string body)
        {
            _httpRequestQueue.Enqueue(new DaemonHttpRequest(operationName, url, body));
        }

        private void ReportHttpResult(
            string operationName,
            long responseCode,
            string error,
            string detail,
            bool success)
        {
            _httpResultQueue.Enqueue(new DaemonHttpResult(
                operationName,
                responseCode,
                error,
                detail,
                success));
        }

        private void EnqueueRawHttpWork(DaemonHttpWorkItem workItem)
        {
            ReportHttpRequest(workItem.OperationName, workItem.Url, workItem.Body);

            lock (_httpWorkLock)
            {
                _pendingHttpWork = workItem;
                _hasPendingHttpWork = true;
                EnsureHttpWorkerLocked();
            }

            _httpWorkSignal.Set();
        }

        private void EnsureHttpWorkerLocked()
        {
            if (_httpWorkerThread != null && _httpWorkerThread.IsAlive)
                return;

            _httpWorkerRunning = true;
            _httpWorkerThread = new Thread(HttpWorkerLoop)
            {
                IsBackground = true,
                Name = "ReachyDaemonHttpSoundThread"
            };
#if UNITY_EDITOR
            Interlocked.Increment(ref httpWorkerStartCountForTests);
#endif
            _httpWorkerThread.Start();
        }

        private void StopHttpWorker()
        {
            _httpWorkerRunning = false;
            lock (_httpWorkLock)
            {
                _hasPendingHttpWork = false;
                _httpWorkInFlight = false;
            }

            _httpWorkSignal.Set();

            try
            {
                if (_httpWorkerThread != null && _httpWorkerThread.IsAlive && _httpWorkerThread != Thread.CurrentThread)
                    _httpWorkerThread.Join(1000);
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[ReachyDaemonTargetWebSocketClient] HTTP worker stop error: {ex.Message}");
            }
            finally
            {
                if (_httpWorkerThread != null && !_httpWorkerThread.IsAlive)
                    _httpWorkerThread = null;
            }
        }

        private void HttpWorkerLoop()
        {
            while (_httpWorkerRunning)
            {
                _httpWorkSignal.WaitOne();

                while (_httpWorkerRunning && TryTakePendingHttpWork(out DaemonHttpWorkItem workItem))
                {
                    int timeoutMilliseconds = Math.Max(1, soundRequestTimeoutSeconds) * 1000;
                    lock (_httpWorkLock)
                    {
                        _httpWorkInFlight = true;
                    }

                    try
                    {
                        RawHttpResult result = ExecuteRawHttpPost(workItem.Uri, workItem.Body, timeoutMilliseconds);
                        ReportHttpResult(
                            workItem.OperationName,
                            result.ResponseCode,
                            result.Error,
                            result.Body,
                            result.Success);
                    }
                    finally
                    {
                        lock (_httpWorkLock)
                        {
                            _httpWorkInFlight = false;
                        }
                    }
                }
            }
        }

        private bool TryTakePendingHttpWork(out DaemonHttpWorkItem workItem)
        {
            lock (_httpWorkLock)
            {
                if (!_hasPendingHttpWork)
                {
                    workItem = default;
                    return false;
                }

                workItem = _pendingHttpWork;
                _pendingHttpWork = default;
                _hasPendingHttpWork = false;
                return true;
            }
        }

        private RawHttpResult ExecuteRawHttpPost(Uri uri, string body, int timeoutMilliseconds)
        {
#if UNITY_EDITOR
            int activeCount = Interlocked.Increment(ref _rawHttpPostActiveCountForTests);
            Interlocked.Increment(ref rawHttpPostStartCountForTests);
            int currentMax;
            while (activeCount > (currentMax = rawHttpPostMaxConcurrentForTests))
                Interlocked.CompareExchange(ref rawHttpPostMaxConcurrentForTests, activeCount, currentMax);
#endif

            try
            {
#if UNITY_EDITOR
                if (rawHttpResponseCodeOverrideForTests != null)
                {
                    long responseCode = rawHttpResponseCodeOverrideForTests(uri, body, timeoutMilliseconds);
                    bool success = responseCode >= 200 && responseCode <= 299;
                    return new RawHttpResult(responseCode, success ? string.Empty : $"HTTP {responseCode}", string.Empty, success);
                }
#endif
                return SendRawHttpPost(uri, body, timeoutMilliseconds);
            }
            finally
            {
#if UNITY_EDITOR
                Interlocked.Decrement(ref _rawHttpPostActiveCountForTests);
#endif
            }
        }

        private static RawHttpResult SendRawHttpPost(Uri uri, string body, int timeoutMilliseconds)
        {
            try
            {
                int port = uri.Port > 0 ? uri.Port : 80;
                byte[] bodyBytes = Encoding.UTF8.GetBytes(body ?? string.Empty);

                using (var client = new TcpClient())
                {
                    var connectTask = client.ConnectAsync(uri.Host, port);
                    if (!connectTask.Wait(timeoutMilliseconds))
                        return RawHttpResult.Failed(0, "Connection timeout", string.Empty);

                    using (NetworkStream stream = client.GetStream())
                    {
                        stream.ReadTimeout = timeoutMilliseconds;
                        stream.WriteTimeout = timeoutMilliseconds;

                        string hostHeader = uri.IsDefaultPort ? uri.Host : $"{uri.Host}:{uri.Port}";
                        string path = string.IsNullOrEmpty(uri.PathAndQuery) ? "/" : uri.PathAndQuery;
                        var header = new StringBuilder();
                        header.Append("POST ").Append(path).Append(" HTTP/1.1\r\n");
                        header.Append("Host: ").Append(hostHeader).Append("\r\n");
                        header.Append("Content-Length: ").Append(bodyBytes.Length).Append("\r\n");
                        if (bodyBytes.Length > 0)
                            header.Append("Content-Type: application/json\r\n");
                        header.Append("Connection: close\r\n");
                        header.Append("\r\n");

                        byte[] headerBytes = Encoding.ASCII.GetBytes(header.ToString());
                        stream.Write(headerBytes, 0, headerBytes.Length);
                        if (bodyBytes.Length > 0)
                            stream.Write(bodyBytes, 0, bodyBytes.Length);
                        stream.Flush();

                        using (var reader = new StreamReader(stream, Encoding.UTF8))
                        {
                            string response = reader.ReadToEnd();
                            return ParseRawHttpResponse(response);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                return RawHttpResult.Failed(0, ex.Message, string.Empty);
            }
        }

        private static RawHttpResult ParseRawHttpResponse(string response)
        {
            if (string.IsNullOrEmpty(response))
                return RawHttpResult.Failed(0, "Empty HTTP response", string.Empty);

            int statusLineEnd = response.IndexOf("\r\n", StringComparison.Ordinal);
            string statusLine = statusLineEnd >= 0 ? response.Substring(0, statusLineEnd) : response;
            string[] statusParts = statusLine.Split(' ');
            long responseCode = 0;
            if (statusParts.Length >= 2)
                long.TryParse(statusParts[1], out responseCode);

            int bodyStart = response.IndexOf("\r\n\r\n", StringComparison.Ordinal);
            string responseBody = bodyStart >= 0 ? response.Substring(bodyStart + 4) : string.Empty;
            bool success = responseCode >= 200 && responseCode <= 299;
            string error = success ? string.Empty : statusLine;
            return new RawHttpResult(responseCode, error, responseBody, success);
        }

        [Serializable]
        private sealed class PlaySoundRequest
        {
            public string file;

            public PlaySoundRequest(string file)
            {
                this.file = file;
            }
        }

        private readonly struct MainThreadLog
        {
            public readonly string message;
            public readonly bool isWarning;

            public MainThreadLog(string message, bool isWarning)
            {
                this.message = message;
                this.isWarning = isWarning;
            }
        }

        private readonly struct DaemonHttpWorkItem
        {
            public readonly string OperationName;
            public readonly string Url;
            public readonly Uri Uri;
            public readonly string Body;

            public DaemonHttpWorkItem(string operationName, string url, Uri uri, string body)
            {
                OperationName = operationName;
                Url = url;
                Uri = uri;
                Body = body;
            }
        }

        internal readonly struct RawHttpResult
        {
            public readonly long ResponseCode;
            public readonly string Error;
            public readonly string Body;
            public readonly bool Success;

            public RawHttpResult(long responseCode, string error, string body, bool success)
            {
                ResponseCode = responseCode;
                Error = error;
                Body = body;
                Success = success;
            }

            public static RawHttpResult Failed(long responseCode, string error, string body)
            {
                return new RawHttpResult(responseCode, error, body, false);
            }
        }

        public readonly struct DaemonHttpRequest
        {
            public readonly string OperationName;
            public readonly string Url;
            public readonly string Body;

            public DaemonHttpRequest(string operationName, string url, string body)
            {
                OperationName = operationName;
                Url = url;
                Body = body;
            }

            public string StatusMessage => FormatHttpRequestStatus(OperationName, Url);
        }

        public readonly struct DaemonHttpResult
        {
            public readonly string OperationName;
            public readonly long ResponseCode;
            public readonly string Error;
            public readonly string Detail;
            public readonly bool Success;

            public DaemonHttpResult(
                string operationName,
                long responseCode,
                string error,
                string detail,
                bool success)
            {
                OperationName = operationName;
                ResponseCode = responseCode;
                Error = error;
                Detail = detail;
                Success = success;
            }

            public string StatusMessage =>
                FormatHttpResultStatus(OperationName, ResponseCode, Error, Detail, Success);
        }
    }
}
