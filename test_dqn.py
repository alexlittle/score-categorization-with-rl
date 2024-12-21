import json
import os
import torch
import pandas as pd
import argparse
import importlib

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from dqn_model.dqn import DQN
from gyms import helper



def get_true_next_score(config, index, current_user_data):
    try:
        next_score = helper.categorize_score(current_user_data.iloc[index].score,
                                             config['categories_range_start'],
                                             config['categories_range_end'],
                                             config['grade_boundaries'])
    except IndexError:
        next_score = -1
    return next_score

def get_predicted_next_score(model, learner_sequence):
    state_tensor = torch.tensor(learner_sequence, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():  # Disable gradient calculation since we are only inferring
        q_values = model(state_tensor)  # Get Q-values from the model

    # Select the action with the highest Q-value (for DQN)
    if q_values.dim() == 1:
        q_values = q_values.unsqueeze(0)
    #print("Q-values:", q_values.numpy())
    action = torch.argmax(q_values, dim=1).item()

    return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file',
                        type=str,
                        help="Data file to use",
                        required=True)
    parser.add_argument('--model',
                        type=str,
                        help="Model to use",
                        required=True)
    parser.add_argument('--simulator',
                        type=str,
                        help="Simulator library to use",
                        required=True)
    args = parser.parse_args()


    try:
        simulator = importlib.import_module(args.simulator)
    except ModuleNotFoundError:
        print(f"Error: The simulator '{args.simulator}' could not be found.")
        return

    simulator_name = simulator.LearningPredictorEnv.__module__.split('.')[1]


    # set up paths to input files
    dir_path = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(dir_path, 'data', args.data_file)
    state_dict_path = os.path.join(dir_path, 'output', 'dqn', simulator_name, args.model + "-model.pth")
    config_path = os.path.join(dir_path, 'output', 'dqn', simulator_name, args.model + "-config.json")
    results_dir = os.path.join(dir_path, 'output', 'dqn', simulator_name, "results")

    # load config
    with open(config_path) as f:
        config = json.load(f)

    # only used here to get access to the normalise_sequence method
    env = simulator.LearningPredictorEnv(data_file_path, config)

    # load model and DQN
    model = DQN(input_size=env.max_sequence_length,
                     hidden_dims=config['hidden_dims'],
                     output_size=config['num_categories']+1)
    model.load_state_dict(torch.load(state_dict_path, weights_only=False))
    model.eval()

    # load data, split into users
    activity = pd.read_csv(data_file_path)
    all_users = activity['id_student'].unique()

    num_exact_correct = 0
    num_exact_incorrect = 0

    num_close_correct = 0
    num_close_incorrect = 0

    # retaining all results
    results = []


    # loop through users until no more activity
    for user in all_users:
        current_user_data = activity.loc[activity['id_student'] == user].sort_values(by='date_submitted')

        #env.current_user_id = user
        #env.learner_demographics = env.init_demographics()
        first_activities = current_user_data.iloc[0].total_vle_before_assessment
        first_score = helper.categorize_score(current_user_data.iloc[0].score,
                                              config['categories_range_start'],
                                              config['categories_range_end'],
                                              config['grade_boundaries'])
        try:
           second_activities = current_user_data.iloc[1].total_vle_before_assessment
        except IndexError:
            second_activities = 0
        learner_sequence = [first_activities, first_score, second_activities]
        #learner_sequence = [first_score]
        for i in range(1, env.max_sequence_length):
            env.learner_sequence = learner_sequence
            normalised_sequence = env.get_observation()

            # check if suggested action matches the actual one
            actual_next_score_category = get_true_next_score(config, i, current_user_data)
            predicted_next_score = get_predicted_next_score(model, normalised_sequence)

            results.append([user, i+1, actual_next_score_category, predicted_next_score-1, abs(actual_next_score_category - (predicted_next_score-1))])

            if actual_next_score_category == -1:
                if predicted_next_score == 0:
                    num_exact_correct += 1
                else:
                    num_exact_incorrect += 1
                if abs(predicted_next_score-1) < 2:
                    num_close_correct += 1
                else:
                    num_close_incorrect += 1
                break

            learner_sequence.append(actual_next_score_category)


            if actual_next_score_category == predicted_next_score-1:
                num_exact_correct += 1
            else:
                num_exact_incorrect += 1
            if abs(actual_next_score_category - (predicted_next_score-1)) <= 1:
                num_close_correct += 1
            else:
                num_close_incorrect += 1

        # output final sequence
        print(learner_sequence)

    results_df = pd.DataFrame(results, columns=["user_id", "assessment_no", "actual_category", "predicted_category", "abs_difference"])

    # save the whole results to csv
    results_df.to_csv(os.path.join(results_dir, "actual_v_predicted.csv"), index=False)

    # create a confusion matrix of the results
    labels = sorted(results_df["actual_category"].unique())  # Ensure labels are sorted
    cm = confusion_matrix(results_df["actual_category"], results_df["predicted_category"], labels=labels)

    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Category")
    plt.ylabel("Actual Category")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")

    # generate and save overall results
    expected_from_random = 100/(config['num_categories']+1)
    actual_exact = num_exact_correct*100 / (num_exact_correct+num_exact_incorrect)
    actual_close = num_close_correct * 100 / (num_close_correct + num_close_incorrect)

    overall_results = {
            'model': args.model,
            'simulator': args.simulator,
            'data_file': args.data_file,
            'expected_from_random': expected_from_random,
            'num_exact_correct': num_exact_correct,
            'num_exact_incorrect': num_exact_incorrect,
            'exact_percent': actual_exact,
            'num_close_correct': num_close_correct,
            'num_close_incorrect': num_close_incorrect,
            'close_percent': actual_close
    }

    results_output_file = os.path.join(results_dir, 'overall.json')
    with open(results_output_file, "w") as file:
        json.dump(overall_results, file, indent=4)


    print(f"random {expected_from_random:.2f}, actual exact: {actual_exact:.2f}, actual close {actual_close:.2f}")
    print(f"Exactly num correct {num_exact_correct}, num incorrect: {num_exact_incorrect}")
    print(f"Close(+/-1) num correct {num_close_correct}, num incorrect: {num_close_incorrect}")


if __name__ == "__main__":
    main()