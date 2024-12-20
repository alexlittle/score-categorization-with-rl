
import pandas as pd
import matplotlib.pyplot as plt


filename = "course_bbb_2013b-train.csv"
# load data file
df = pd.read_csv(filename)


# total no students
num_students = len(df['id_student'].unique())
print(f"Num students: {num_students}")

# total no records
count_rows = df.shape[0]
print(count_rows)


# histogram number of assessments students have completed
record_counts = df.groupby('id_student').size()

# Plot the histogram
plt.figure(figsize=(8, 6))
plt.hist(record_counts, bins=range(1, record_counts.max() + 2), align='left', edgecolor='black')
plt.title('Histogram of Number of Assessments completed')
plt.xlabel('Number of Records')
plt.ylabel('Number of Students')
plt.xticks(range(1, record_counts.max() + 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("completion-histogram.png", dpi=300, bbox_inches="tight")


# histogram assessment scores
record_counts = df.groupby('score').size()

print(record_counts)
print(record_counts.max())
# Plot the histogram
plt.figure(figsize=(8, 6))
plt.hist(record_counts, bins=range(1, record_counts.max() + 2), align='left', edgecolor='black')
plt.title('Histogram of Assessment Scores')
plt.xlabel('Score')
plt.ylabel('Number of Assessments')
plt.xticks(range(1, record_counts.max() + 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("score-histogram.png", dpi=300, bbox_inches="tight")