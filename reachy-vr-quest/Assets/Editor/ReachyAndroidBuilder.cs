using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Android;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace ReachyMiniTeleop.Editor
{
    public static class ReachyAndroidBuilder
    {
        public static void Build()
        {
            string outputPath = Path.GetFullPath(GetArg("-apkPath") ?? "Builds/ReachyMiniTeleop.apk");
            string outputDirectory = Path.GetDirectoryName(outputPath);
            if (!string.IsNullOrEmpty(outputDirectory))
                Directory.CreateDirectory(outputDirectory);

            ConfigureAndroidTools();

            string[] scenes = EditorBuildSettings.scenes
                .Where(scene => scene.enabled)
                .Select(scene => scene.path)
                .ToArray();
            if (scenes.Length == 0)
                throw new InvalidOperationException("No enabled scenes in EditorBuildSettings.");

            Debug.Log($"[ReachyAndroidBuilder] Building {outputPath} from {string.Join(", ", scenes)}");

            var options = new BuildPlayerOptions
            {
                scenes = scenes,
                locationPathName = outputPath,
                target = BuildTarget.Android,
                targetGroup = BuildTargetGroup.Android,
                options = BuildOptions.None
            };

            BuildReport report = BuildPipeline.BuildPlayer(options);
            BuildSummary summary = report.summary;
            if (summary.result != BuildResult.Succeeded)
            {
                Debug.LogError(
                    $"[ReachyAndroidBuilder] Build {summary.result}. errors={summary.totalErrors} output={outputPath}");
                EditorApplication.Exit(1);
                return;
            }

            Debug.Log($"[ReachyAndroidBuilder] APK ready: {outputPath} ({summary.totalSize} bytes)");
        }

        private static void ConfigureAndroidTools()
        {
            string playerRoot = Path.GetFullPath(Path.Combine(
                EditorApplication.applicationContentsPath,
                "PlaybackEngines/AndroidPlayer"));
            TrySetSdk(Path.Combine(playerRoot, "SDK"));
            TrySetNdk(Path.Combine(playerRoot, "NDK"));
            TrySetJdk(Path.Combine(playerRoot, "OpenJDK"));
        }

        private static void TrySetSdk(string path)
        {
            if (!Directory.Exists(path))
            {
                Debug.LogWarning($"[ReachyAndroidBuilder] Missing Android SDK: {path}");
                return;
            }

            AndroidExternalToolsSettings.sdkRootPath = path;
            Debug.Log($"[ReachyAndroidBuilder] Android SDK={path}");
        }

        private static void TrySetNdk(string path)
        {
            if (!Directory.Exists(path))
            {
                Debug.LogWarning($"[ReachyAndroidBuilder] Missing Android NDK: {path}");
                return;
            }

            AndroidExternalToolsSettings.ndkRootPath = path;
            Debug.Log($"[ReachyAndroidBuilder] Android NDK={path}");
        }

        private static void TrySetJdk(string path)
        {
            if (!Directory.Exists(path))
            {
                Debug.LogWarning($"[ReachyAndroidBuilder] Missing OpenJDK: {path}");
                return;
            }

            AndroidExternalToolsSettings.jdkRootPath = path;
            Debug.Log($"[ReachyAndroidBuilder] OpenJDK={path}");
        }

        private static string GetArg(string name)
        {
            string[] args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
            {
                if (args[i] == name)
                    return args[i + 1];
            }

            return null;
        }
    }
}
