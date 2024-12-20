import gymnasium
from gymnasium import spaces
import numpy as np
import pandas as pd
import random

from gyms import helper

class LearningPredictorEnv(gymnasium.Env):

    def __init__(self,  data_file_path, config):
        super(LearningPredictorEnv, self).__init__()

        self.max_sequence_length = 11
        self.config = config
        self.all_activity = pd.read_csv(data_file_path)
        self.all_users = self.all_activity['id_student'].unique()

        self.grade_boundaries = self.config['grade_boundaries']
        self.num_categories = self.config['num_categories']
        self.observation_space = spaces.Box(low=0, high=self.max_sequence_length, shape=(1,), dtype=np.int32)
        self.action_space = spaces.Discrete(self.num_categories+1)
        self.learner_sequence = []
        self.current_user_id = 0
        self.current_user_data = []
        self.current_user_data_index = 0

    def reset(self):
        self.current_user_data_index = 0
        #self.current_user_id = 2569163 - user with all 11 assessments completed
        # expected: 81, 81, 100, 83, 100, 87, 80, 83, 100, 89, 80
        self.current_user_id = random.choice(self.all_users)
        self.current_user_data = self.all_activity.loc[self.all_activity['id_student'] == self.current_user_id].sort_values(by='date_submitted')

        # add first assessment score
        first_score = helper.categorize_score(self.current_user_data.iloc[self.current_user_data_index].score,
                                              self.config['categories_range_start'],
                                              self.config['categories_range_end'],
                                              self.config['grade_boundaries'])

        self.learner_sequence = [first_score]

        return self.get_observation()

    def step(self, action):
        true_next_score_category = self._get_true_next_category()
        if true_next_score_category == -1:
            if action == 0: # has correctly predicted end of learner activity
                return self.get_observation(), 1, True, {}
            else:
                return self.get_observation(), 0, True, {}
        self.learner_sequence.append(true_next_score_category)

        reward = 1 if action-1 == true_next_score_category else 0
        # 5. Check for episode termination (adjust as needed)
        done = len(self.learner_sequence) >= self.max_sequence_length

        return self.get_observation(), reward, done, {}

    def get_observation(self):
        padded_arr = self.learner_sequence + [-1] * (self.max_sequence_length - len(self.learner_sequence))
        for idx, x in enumerate(padded_arr):
            if x != -1:
                padded_arr[idx] = x / (self.num_categories - 1)
        return padded_arr

    def _get_true_next_category(self):
        self.current_user_data_index += 1
        try:
            next_score = helper.categorize_score(self.current_user_data.iloc[self.current_user_data_index].score,
                                                 range_start = self.config['categories_range_start'],
                                                 range_end = self.config['categories_range_end'],
                                                 step = self.config['grade_boundaries'])
        except IndexError:
            next_score = -1

        return next_score


