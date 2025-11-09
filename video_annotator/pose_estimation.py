import mediapipe as mp
import cv2
import numpy as np


class PoseEstimator:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.pose_hands_video = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def estimate_pose_video(self, video_path: str):
        video_cap = cv2.VideoCapture(video_path)
        pose_estimates = []
        frame_index = 0

        while video_cap.isOpened():
            success, frame = video_cap.read()
            if not success:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False

            frame_pose_estimates = self.pose_hands_video.process(frame)
            timestamp = video_cap.get(cv2.CAP_PROP_POS_MSEC)

            frame_record = {
                "frame_index": frame_index,
                "hands": []
            }

            if frame_pose_estimates.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(frame_pose_estimates.multi_hand_landmarks):
                    hand_label = None

                    if frame_pose_estimates.multi_handedness:
                        hand_label = frame_pose_estimates.multi_handedness[hand_idx].classification[0].label

                    joints = {}
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        joint_name = self.mp_hands.HandLandmark(i).name
                        joints[joint_name] = {
                            "x": landmark.x,
                            "y": landmark.y,
                            "z": landmark.z,
                        }

                    frame_record["hands"].append({
                        "handedness": hand_label,
                        "joints": joints
                    })

            pose_estimates.append(frame_record)
            frame_index += 1

        video_cap.release()
        return pose_estimates


if __name__ == "__main__":
    pose_estimator = PoseEstimator()
    pose_estimates = pose_estimator.estimate_pose_video("video_data/0.mp4")
    print(pose_estimates[29])
