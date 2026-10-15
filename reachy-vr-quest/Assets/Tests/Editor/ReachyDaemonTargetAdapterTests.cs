using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using ReachyMiniTeleop.Reachy;
using UnityEngine;

namespace ReachyMiniTeleop.Tests.Editor
{
    public sealed class ReachyDaemonTargetAdapterTests
    {
        [Test]
        public void FromHeadPayload_ComposesBodyYawAndLocalHeadYawIntoWorldHeadPose()
        {
            var payload = new ReachyHeadPayload
            {
                bodyYawDegrees = 30f,
                headPosition = new ReachyVectorPayload(0f, 0f, 0f),
                headRotation = new ReachyQuaternionPayload(Quaternion.AngleAxis(10f, Vector3.forward)),
                antennas = new ReachyAntennaPayload { right = 0.1f, left = -0.2f }
            };

            ReachyDaemonFullBodyTarget target = ReachyDaemonTargetAdapter.FromHeadPayload(payload);

            Assert.AreEqual(Mathf.Deg2Rad * 30f, target.targetBodyYaw.Value, 0.0001f);
            Assert.AreEqual(0.1f, target.targetAntennas[0], 0.0001f);
            Assert.AreEqual(-0.2f, target.targetAntennas[1], 0.0001f);
            Assert.AreEqual(40f, ExtractYawDegrees(target.targetHeadPose.m), 0.001f);
        }

        [Test]
        public void FromHeadPayload_WhenBodyYawIsNull_ComposesWithPayloadYawAndOmitsTargetBodyYaw()
        {
            var payload = new ReachyHeadPayload
            {
                bodyYawDegrees = 30f,
                headPosition = new ReachyVectorPayload(0f, 0f, 0f),
                headRotation = new ReachyQuaternionPayload(Quaternion.AngleAxis(10f, Vector3.forward)),
                antennas = new ReachyAntennaPayload { right = 0f, left = 0f }
            };

            ReachyDaemonFullBodyTarget target = ReachyDaemonTargetAdapter.FromHeadPayload(payload, null);
            string json = JsonConvert.SerializeObject(target, new JsonSerializerSettings
            {
                NullValueHandling = NullValueHandling.Ignore
            });
            JObject parsed = JObject.Parse(json);

            Assert.IsFalse(target.targetBodyYaw.HasValue);
            Assert.IsNull(parsed["target_body_yaw"]);
            Assert.AreEqual(40f, ExtractYawDegrees(target.targetHeadPose.m), 0.001f);
        }

        [Test]
        public void ToJson_IncludesArmPoses()
        {
            var payload = new ReachyHeadPayload
            {
                bodyYawDegrees = 0f,
                headPosition = new ReachyVectorPayload(0f, 0f, 0f),
                headRotation = new ReachyQuaternionPayload(Quaternion.identity),
                antennas = new ReachyAntennaPayload(),
                arms = new ReachyArmsPayload
                {
                    left = new ReachyArmPayload
                    {
                        wrist = new ReachyJointPosePayload
                        {
                            position = new ReachyVectorPayload(0.1f, 0.2f, 0.3f),
                            rotation = new ReachyQuaternionPayload(Quaternion.identity)
                        }
                    }
                }
            };

            JObject parsed = JObject.Parse(ReachyDaemonTargetAdapter.ToJson(payload));

            Assert.AreEqual(0.1f, (float)parsed["arms"]["left"]["wrist"]["position"]["x"], 0.0001f);
            Assert.AreEqual(0.2f, (float)parsed["arms"]["left"]["wrist"]["position"]["y"], 0.0001f);
            Assert.AreEqual(0.3f, (float)parsed["arms"]["left"]["wrist"]["position"]["z"], 0.0001f);
        }

        [Test]
        public void ToJson_IncludesControllerButtons()
        {
            var payload = new ReachyHeadPayload
            {
                bodyYawDegrees = 0f,
                headPosition = new ReachyVectorPayload(0f, 0f, 0f),
                headRotation = new ReachyQuaternionPayload(Quaternion.identity),
                antennas = new ReachyAntennaPayload(),
                controllers = new ReachyControllersPayload
                {
                    left = new ReachyControllerPayload
                    {
                        connected = true,
                        trigger = 0.8f,
                        grip = 0.25f,
                        primary = true,
                        secondary = false,
                        thumbstickClick = true,
                        menu = false,
                        thumbstick = new ReachyStickPayload { x = -0.4f, y = 0.6f }
                    }
                }
            };

            JObject parsed = JObject.Parse(ReachyDaemonTargetAdapter.ToJson(payload));
            JToken left = parsed["controllers"]["left"];

            Assert.IsTrue((bool)left["connected"]);
            Assert.AreEqual(0.8f, (float)left["trigger"], 0.0001f);
            Assert.AreEqual(0.25f, (float)left["grip"], 0.0001f);
            Assert.IsTrue((bool)left["primary"]);
            Assert.IsFalse((bool)left["secondary"]);
            Assert.IsTrue((bool)left["thumbstick_click"]);
            Assert.IsFalse((bool)left["menu"]);
            Assert.AreEqual(-0.4f, (float)left["thumbstick"]["x"], 0.0001f);
            Assert.AreEqual(0.6f, (float)left["thumbstick"]["y"], 0.0001f);
        }

        [Test]
        public void FlattenRowMajor_UsesDaemonMatrixOrder()
        {
            Matrix4x4 matrix = Matrix4x4.identity;
            matrix[0, 1] = 2f;
            matrix[2, 3] = 5f;

            float[] values = ReachyDaemonTargetAdapter.FlattenRowMajor(matrix);

            Assert.AreEqual(2f, values[1], 0.001f);
            Assert.AreEqual(5f, values[11], 0.001f);
        }

        private static float ExtractYawDegrees(float[] rowMajorMatrix)
        {
            return Mathf.Atan2(rowMajorMatrix[4], rowMajorMatrix[0]) * Mathf.Rad2Deg;
        }
    }
}
