import os
import time
import pybullet as bullet
import imageio
import numpy as np
from pybullet_envs.bullet import KukaGymEnv
from pybullet_envs.bullet import kuka
from gym import spaces
import pybullet_data
from PIL import Image

import cv2

class KukaImitatorEnv(KukaGymEnv):
    # The goal in this env is to get the robot in a given pose.
    def __init__(self, urdfRoot=pybullet_data.getDataPath(), actionRepeat=1, video_directory=None, model=None, isEnableSelfCollision=True, renders=True):
        self._timeStep = 1./240.
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = None
        self._envStepCounter = 0
        self._renders = renders
        self.terminated = 0
        self._setup_rendering()
        self._setup_kuka()
        self._setup_spaces()
        self._img_state = None
        self._seed()

        self.FRAME_ID = None
        # TCN 
        self.model = model 
        self.videos = [self._read_video(video_directory, v) for v in os.listdir(video_directory)]
        self.which_video = 0 
        self.video_length = len(self.videos[self.which_video])

        # capturing the image
        self.pixel_height = 1000 
        self.pixel_width = 900 
        self.near_plane = 0.01
        self.far_plane = 100
        self.fov = 60
        self.camera_position = {
                'distance': 3,
                'pitch': -10,
                'yaw': 90,
            }
        self.reset()

    def _process_frames(self, *frames):
        resized = []
        for frame in frames:
            image = Image.fromarray(frame)
            image = image.resize((299, 299))
            resized.append(np.array(image) / 255)
        
        #cv2.imwrite('./imitator/states/{}-frame.png'.format(self._envStepCounter), resized[0])
        
        stacked = np.stack(resized).astype(np.float32)
        return np.transpose(stacked, [0, 3, 1, 2])

    def _read_video(self, path, video):
        path = os.path.join(path, video)
        return imageio.read(path) 
    
    def _get_target_frame(self, frame_index):
        video = self.videos[self.which_video]
        frame = video.get_data(frame_index)
        return self._process_frames(frame)

    def _setup_rendering(self):
        if self._renders:
            cid = bullet.connect(bullet.SHARED_MEMORY)
            if (cid<0):
                cid = bullet.connect(bullet.GUI)
            bullet.resetDebugVisualizerCamera(1.3,180,-41,[0.52,-0.2,-0.33])
        else:
            bullet.connect(bullet.DIRECT)

    def _setup_kuka(self):
        self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._kuka.useInverseKinematics = False

    def _setup_spaces(self):
        action_dimensions = len(self._kuka.motorIndices)
        self.action_space = spaces.Box(
            low=-np.ones(action_dimensions) * self._kuka.maxForce,
            high=np.ones(action_dimensions) * self._kuka.maxForce)

        joint_lower_limit = np.zeros(self._kuka.numJoints)
        joint_upper_limit = np.zeros(self._kuka.numJoints)
        for joint_index in range(self._kuka.numJoints):
            joint_info = bullet.getJointInfo(self._kuka.kukaUid, joint_index)
            joint_lower_limit[joint_index] = joint_info[8]
            joint_upper_limit[joint_index] = joint_info[9]

        self.observation_space = spaces.Box(
            low=joint_lower_limit,
            high=joint_upper_limit)

    def _reset(self):
        self.terminated = 0
        bullet.resetSimulation()
        bullet.setPhysicsEngineParameter(numSolverIterations=150)
        bullet.setTimeStep(self._timeStep)
        bullet.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"),[0, 0, 0])
        bullet.setGravity(0, 0, -10)


        

        # define target video
        self.which_video = 0
        self.video_length = len(self.videos[self.which_video])

        self._kuka.reset()
        self._envStepCounter = 0
        bullet.stepSimulation()
        self._observation = self.getExtendedObservation()
        self.joint_history = [self._observation]

        return self._observation

    def _termination(self):
        return not(self._envStepCounter < self.video_length)

    def _step(self, action):
        self._kuka.applyAction(action)
        for i in range(self._actionRepeat):
            bullet.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)
            self._observation = self.getExtendedObservation()
            done = self._termination()
            if done:
                break
            self._envStepCounter += 1
        reward = self._reward()
        return self._observation, reward, done, {}

    def currentFrame(self):
        for i in range(bullet.getNumBodies()):
            if (bullet.getBodyInfo(i)[0].decode() == "lbr_iiwa_link_0"):
                self.FRAME_ID = i
        assert(not(self.FRAME_ID is None))
        # camera position
        human_pos, humanOrn = bullet.getBasePositionAndOrientation(self.FRAME_ID)
        distance = self.camera_position['distance']
        yaw = self.camera_position['yaw']
        pitch = self.camera_position['pitch']

        view_matrix = bullet.computeViewMatrixFromYawPitchRoll(human_pos, distance, yaw, pitch, 0, 2)

        # projection matrix
        aspect = self.pixel_width / self.pixel_height
        projection_matrix = bullet.computeProjectionMatrixFOV(self.fov, aspect, self.near_plane, self.far_plane)

        # image
        w, h, rgb_pixels, depth, segmentation  = bullet.getCameraImage(self.pixel_width, self.pixel_height, view_matrix, projection_matrix, shadow=1, lightDirection=[1,1,1],renderer=bullet.ER_BULLET_HARDWARE_OPENGL)
        
        rgb_pixels = rgb_pixels[:, :, :3] 
        # cv2.imwrite('./imitator/states/{}-state.png'.format(self._envStepCounter), rgb_pixels)

        return self._process_frames(rgb_pixels)

    def _reward(self):
        
        current_frame = self.currentFrame()
        current_state = self.model(current_frame)

        target_state = self.model(self._get_target_frame(self._envStepCounter))
        distance = - np.linalg.norm(np.abs(target_state - current_state), 2) ** 2
        assert(distance <= 0)

        return distance

    def getExtendedObservation(self):
        joint_states = bullet.getJointStates(self._kuka.kukaUid, range(self._kuka.numJoints))
        # The first item in the joint state is the joint position.
        joint_positions = [jointState[0] for jointState in joint_states]
        return np.array(joint_positions)

if __name__ == '__main__':
    main()