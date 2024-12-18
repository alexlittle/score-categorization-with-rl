import gymnasium
from gymnasium import spaces
import numpy as np
import pandas as pd
import random

class LearningPredictorEnv(gymnasium.Env):
    def __init__(self,  data_file_path, max_sequence_length=11, grade_boundaries=10, num_categories=6):
        super(LearningPredictorEnv, self).__init__()


        activity = pd.read_csv(data_file_path)

        # filter for only one courses assessments
        self.all_activity = activity
        self.all_users = self.all_activity['id_student'].unique()
        self.max_sequence_length = max_sequence_length
        self.grade_boundaries = grade_boundaries
        self.num_categories = num_categories
        self.observation_space = spaces.Box(low=0, high=self.max_sequence_length, shape=(1,), dtype=np.int32)
        self.action_space = spaces.Discrete(self.num_categories+1)
        self.learner_sequence = []
        self.current_user_id = 0
        self.current_user_data = []
        self.current_user_data_index = 0


    def reset(self):
        self.current_user_data_index = 0
        self.current_user_id = random.choice(self.all_users)
        self.current_user_data = self.all_activity.loc[self.all_activity['id_student'] == self.current_user_id].sort_values(by='date_submitted')

        # add first assessment score
        first_score = self.categorize_score(self.current_user_data.iloc[self.current_user_data_index].score)

        self.learner_sequence = [first_score]

        return self._get_observation()

    def step(self, action):
        true_next_score_category = self._get_true_next_category()
        if true_next_score_category == -1:
            if action == 0: # has correctly predicted end of learner activity
                return self._get_observation(), 1, True, {}
            else:
                return self._get_observation(), 0, True, {}
        self.learner_sequence.append(true_next_score_category)

        reward = 1 if action-1 == true_next_score_category else 0
        # 5. Check for episode termination (adjust as needed)
        done = len(self.learner_sequence) > self.max_sequence_length

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        padded_arr = self.learner_sequence + [-1] * (self.max_sequence_length - len(self.learner_sequence))
        for idx, x in enumerate(padded_arr):
            if x != -1:
                padded_arr[idx] = x / self.num_categories
        return padded_arr

    def _get_true_next_category(self):
        self.current_user_data_index += 1
        try:
            next_score = self.categorize_score(self.current_user_data.iloc[self.current_user_data_index].score)
        except IndexError:
            next_score = -1

        return next_score

    def categorize_score(self, score):
        for idx, x in enumerate(range(50, 91, self.grade_boundaries)):
            if score < x:
                return idx
        return idx + 1

