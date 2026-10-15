using Newtonsoft.Json;
using UnityEngine;

namespace ReachyMiniTeleop.Reachy
{
    [System.Serializable]
    public sealed class ReachyVectorPayload
    {
        public float x;
        public float y;
        public float z;

        public ReachyVectorPayload()
        {
        }

        public ReachyVectorPayload(float x, float y, float z)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }
    }

    [System.Serializable]
    public sealed class ReachyQuaternionPayload
    {
        public float x;
        public float y;
        public float z;
        public float w;

        public ReachyQuaternionPayload()
        {
        }

        public ReachyQuaternionPayload(Quaternion rotation)
        {
            x = rotation.x;
            y = rotation.y;
            z = rotation.z;
            w = rotation.w;
        }

        public Quaternion ToQuaternion()
        {
            return new Quaternion(x, y, z, w);
        }
    }

    [System.Serializable]
    public sealed class ReachyAntennaPayload
    {
        public float right;
        public float left;
    }

    [System.Serializable]
    public sealed class ReachyJointPosePayload
    {
        [JsonProperty("position")]
        public ReachyVectorPayload position;

        [JsonProperty("rotation")]
        public ReachyQuaternionPayload rotation;
    }

    [System.Serializable]
    public sealed class ReachyArmPayload
    {
        [JsonProperty("shoulder", NullValueHandling = NullValueHandling.Ignore)]
        public ReachyJointPosePayload shoulder;

        [JsonProperty("upper_arm", NullValueHandling = NullValueHandling.Ignore)]
        public ReachyJointPosePayload upperArm;

        [JsonProperty("lower_arm", NullValueHandling = NullValueHandling.Ignore)]
        public ReachyJointPosePayload lowerArm;

        [JsonProperty("wrist", NullValueHandling = NullValueHandling.Ignore)]
        public ReachyJointPosePayload wrist;
    }

    [System.Serializable]
    public sealed class ReachyArmsPayload
    {
        [JsonProperty("left", NullValueHandling = NullValueHandling.Ignore)]
        public ReachyArmPayload left;

        [JsonProperty("right", NullValueHandling = NullValueHandling.Ignore)]
        public ReachyArmPayload right;
    }

    [System.Serializable]
    public sealed class ReachyStickPayload
    {
        public float x;
        public float y;
    }

    [System.Serializable]
    public sealed class ReachyControllerPayload
    {
        [JsonProperty("connected")]
        public bool connected;

        [JsonProperty("trigger")]
        public float trigger;

        [JsonProperty("grip")]
        public float grip;

        // Left: X. Right: A.
        [JsonProperty("primary")]
        public bool primary;

        // Left: Y. Right: B.
        [JsonProperty("secondary")]
        public bool secondary;

        [JsonProperty("thumbstick_click")]
        public bool thumbstickClick;

        [JsonProperty("menu")]
        public bool menu;

        [JsonProperty("thumbstick")]
        public ReachyStickPayload thumbstick;

        // Grip pose in the same robot FLU frame as the arm bones.
        [JsonProperty("position", NullValueHandling = NullValueHandling.Ignore)]
        public ReachyVectorPayload position;

        [JsonProperty("rotation", NullValueHandling = NullValueHandling.Ignore)]
        public ReachyQuaternionPayload rotation;
    }

    [System.Serializable]
    public sealed class ReachyControllersPayload
    {
        [JsonProperty("left", NullValueHandling = NullValueHandling.Ignore)]
        public ReachyControllerPayload left;

        [JsonProperty("right", NullValueHandling = NullValueHandling.Ignore)]
        public ReachyControllerPayload right;
    }

    [System.Serializable]
    public sealed class ReachyHeadPayload
    {
        [JsonProperty("body_yaw_degrees")]
        public float bodyYawDegrees;

        [JsonProperty("head_position")]
        public ReachyVectorPayload headPosition;

        [JsonProperty("head_rotation")]
        public ReachyQuaternionPayload headRotation;

        [JsonProperty("antennas")]
        public ReachyAntennaPayload antennas;

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

