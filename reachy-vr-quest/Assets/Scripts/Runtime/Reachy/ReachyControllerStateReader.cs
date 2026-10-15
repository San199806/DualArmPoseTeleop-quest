using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;

namespace ReachyMiniTeleop.Reachy
{
    public static class ReachyControllerStateReader
    {
        private static readonly List<InputDevice> Devices = new List<InputDevice>();

        public static ReachyControllersPayload Capture()
        {
            ReachyControllerPayload left = CaptureOpenXr(XRNode.LeftHand);
            ReachyControllerPayload right = CaptureOpenXr(XRNode.RightHand);
            if (left == null || right == null)
            {
                OVRInput.Update();
                if (left == null)
                    left = CaptureOvr(XRNode.LeftHand);
                if (right == null)
                    right = CaptureOvr(XRNode.RightHand);
            }

            return new ReachyControllersPayload
            {
                left = left ?? Disconnected(),
                right = right ?? Disconnected()
            };
        }

        private static ReachyControllerPayload CaptureOpenXr(XRNode node)
        {
            Devices.Clear();
            InputDeviceCharacteristics side = node == XRNode.LeftHand
                ? InputDeviceCharacteristics.Left
                : InputDeviceCharacteristics.Right;
            InputDevices.GetDevicesWithCharacteristics(
                InputDeviceCharacteristics.Controller | InputDeviceCharacteristics.HeldInHand | side,
                Devices);
            if (Devices.Count == 0)
                InputDevices.GetDevicesAtXRNode(node, Devices);

            for (int i = 0; i < Devices.Count; i++)
            {
                if (TryReadOpenXrDevice(Devices[i], out ReachyControllerPayload payload))
                    return payload;
            }

            return null;
        }

        private static bool TryReadOpenXrDevice(InputDevice device, out ReachyControllerPayload payload)
        {
            payload = null;
            if (!device.isValid)
                return false;

            var candidate = Disconnected();
            bool any = false;
            any |= device.TryGetFeatureValue(CommonUsages.trigger, out candidate.trigger);
            any |= device.TryGetFeatureValue(CommonUsages.grip, out candidate.grip);
            any |= device.TryGetFeatureValue(CommonUsages.primaryButton, out candidate.primary);
            any |= device.TryGetFeatureValue(CommonUsages.secondaryButton, out candidate.secondary);
            any |= device.TryGetFeatureValue(CommonUsages.primary2DAxisClick, out candidate.thumbstickClick);
            any |= device.TryGetFeatureValue(CommonUsages.menuButton, out candidate.menu);
            if (device.TryGetFeatureValue(CommonUsages.primary2DAxis, out Vector2 stick))
            {
                candidate.thumbstick.x = stick.x;
                candidate.thumbstick.y = stick.y;
                any = true;
            }

            bool hasPosition = device.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 position);
            bool hasRotation = device.TryGetFeatureValue(CommonUsages.deviceRotation, out Quaternion rotation);
            if (hasPosition && hasRotation)
            {
                AttachPose(candidate, position, rotation);
                any = true;
            }

            if (!any)
                return false;

            candidate.connected = true;
            payload = candidate;
            return true;
        }

        private static ReachyControllerPayload CaptureOvr(XRNode node)
        {
            if (node != XRNode.LeftHand && node != XRNode.RightHand)
                return null;

            OVRInput.Controller controller = node == XRNode.LeftHand
                ? OVRInput.Controller.LTouch
                : OVRInput.Controller.RTouch;
            if (!OVRInput.IsControllerConnected(controller))
                return null;

            Vector2 stick = OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick, controller);
            var payload = new ReachyControllerPayload
            {
                connected = true,
                trigger = OVRInput.Get(OVRInput.Axis1D.PrimaryIndexTrigger, controller),
                grip = OVRInput.Get(OVRInput.Axis1D.PrimaryHandTrigger, controller),
                primary = OVRInput.Get(OVRInput.Button.One, controller),
                secondary = OVRInput.Get(OVRInput.Button.Two, controller),
                thumbstickClick = OVRInput.Get(OVRInput.Button.PrimaryThumbstick, controller),
                menu = OVRInput.Get(OVRInput.Button.Start, controller),
                thumbstick = new ReachyStickPayload { x = stick.x, y = stick.y }
            };
            AttachPose(
                payload,
                OVRInput.GetLocalControllerPosition(controller),
                OVRInput.GetLocalControllerRotation(controller));
            return payload;
        }

        private static void AttachPose(ReachyControllerPayload payload, Vector3 trackingPosition, Quaternion trackingRotation)
        {
            Vector3 worldPosition = trackingPosition;
            Quaternion worldRotation = trackingRotation;
            OVRCameraRig rig = OVRManager.instance != null
                ? OVRManager.instance.GetComponent<OVRCameraRig>()
                : null;
            if (rig != null && rig.trackingSpace != null)
            {
                worldPosition = rig.trackingSpace.TransformPoint(trackingPosition);
                worldRotation = rig.trackingSpace.rotation * trackingRotation;
            }

            Vector3 positionFlu = CoordinateFrameUtil.UnityRufToRobotFlu(worldPosition);
            Quaternion rotationFlu = CoordinateFrameUtil.UnityRufToRobotFlu(worldRotation);
            payload.position = new ReachyVectorPayload(positionFlu.x, positionFlu.y, positionFlu.z);
            payload.rotation = new ReachyQuaternionPayload(rotationFlu);
        }

        private static ReachyControllerPayload Disconnected()
        {
            return new ReachyControllerPayload
            {
                connected = false,
                thumbstick = new ReachyStickPayload()
            };
        }
    }
}
