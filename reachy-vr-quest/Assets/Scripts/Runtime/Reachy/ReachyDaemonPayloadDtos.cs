using Newtonsoft.Json;

namespace ReachyMiniTeleop.Reachy
{
    [System.Serializable]
    public sealed class ReachyDaemonMatrixPose
    {
        [JsonProperty("m")]
        public float[] m;
    }

    [System.Serializable]
    public sealed class ReachyDaemonFullBodyTarget
    {
        [JsonProperty("target_head_pose", NullValueHandling = NullValueHandling.Ignore)]
        public ReachyDaemonMatrixPose targetHeadPose;

        [JsonProperty("target_antennas", NullValueHandling = NullValueHandling.Ignore)]
        public float[] targetAntennas;

        [JsonProperty("target_body_yaw", NullValueHandling = NullValueHandling.Ignore)]
        public float? targetBodyYaw;

        [JsonProperty("arms", NullValueHandling = NullValueHandling.Ignore)]
        public ReachyArmsPayload arms;

        [JsonProperty("controllers", NullValueHandling = NullValueHandling.Ignore)]
        public ReachyControllersPayload controllers;

        [JsonProperty("hips", NullValueHandling = NullValueHandling.Ignore)]
        public ReachyJointPosePayload hips;

        [JsonProperty("chest", NullValueHandling = NullValueHandling.Ignore)]
        public ReachyJointPosePayload chest;
    }
}
