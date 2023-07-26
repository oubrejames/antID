import pandas as pd
import matplotlib.pyplot as plt
#Upload the data and save it to a Dataframe.
df = pd.read_csv( '../ant_face_data/labels.csv', header=None)
df=pd.DataFrame(df)
print("df size: ", df.shape)

# Check the class distribution. (Hint: Use .value_counts attribute and 
# remember that the labels are at the last column of the dataframe)
results = df[0]

fig1, ax1 = plt.subplots()
ax1.set_title("Class Distribution")
for i in range(len(df[0].value_counts())):
    ax1.bar(i, height=df[0].value_counts()[i], width=0.5)
ax1.set_xlabel('Class')
ax1.set_ylabel("Number of Instances")
fig1.savefig("Class Distribution")

print('Class Distribution: ')
print(df[0].value_counts())

# Ants 11, 12, 16, and 26 have much more instances than the other training data and ant 8 has barely
# any.
# Make the dataset more balanced by sampling ants 11, 12, 16, and 26 and removing ant 8
missing_ants = df.drop(df[df[0] == 'ant_11'])
# The value you want to remove from the DataFrame
target_classes = ["ant_11", "ant_12", "ant_16", "ant_26", "ant_8"]

filtered_df = df
for target_class in target_classes:
    # Boolean indexing to filter out rows with the target_class
    filtered_df = filtered_df[filtered_df[0] != target_class]

# print(missing_ants)
# missing_ants.drop(missing_ants[missing_ants[0] == 'ant_12'])
# missing_ants = missing_ants.drop(missing_ants[missing_ants[0] == 'ant_16'])
# missing_ants = missing_ants.drop(missing_ants[missing_ants[0] == 'ant_26'])
# missing_ants = missing_ants.drop(missing_ants[missing_ants[0] == 'ant_8'])

all_11 = df[df[0] == 'ant_11']
all_12 = df[df[0] == 'ant_12']
all_16 = df[df[0] == 'ant_16']
all_26 = df[df[0] == 'ant_26']

smaller_11 = all_11.sample(860)
smaller_12 = all_12.sample(860)
smaller_16 = all_16.sample(860)
smaller_26 = all_26.sample(860)

# # balanced_df = balanced_df.drop(df[df[0] == 'ant_12'].sample(860).index)
# # balanced_df = balanced_df.drop(df[df[0] == 'ant_16'].sample(860).index)
# # balanced_df = balanced_df.drop(df[df[0] == 'ant_26'].sample(860).index)
# # balanced_df = balanced_df.drop(df[df[0] == 'ant_8'])
balanced_df = pd.concat([smaller_11, smaller_12, smaller_16, smaller_26, filtered_df], axis=0)

fig2, ax2 = plt.subplots()
ax2.set_title("Balanced Class Distribution")
for i in range(len(balanced_df[0].value_counts())):
    ax2.bar(i, height=balanced_df[0].value_counts()[i], width=0.5)
ax2.set_xlabel('Class')
ax2.set_ylabel("Number of Instances")
fig2.savefig("Balanced Class Distribution")

print('Balanced Class Distribution: ')
print(balanced_df[0].value_counts())



# all_zeros_data = data[data[data.shape[1]-1] == 0]

# # From all the data that has class 0 (again use filtering), sample 8000 instances and save it to a new dataframe. 
# #(Hint: Use .sample attribute)
# smaller_zeros_df = all_zeros_data.sample(8000)

# # Concatenate the two new dataframes.
# balanced_df = pd.concat([smaller_zeros_df, no_zero_df], axis=0)

# # Check the new class distribution.
# plt.suptitle("Balanced Class Distribution")
# for i in range(len(balanced_df[balanced_df.shape[1]-1].value_counts())):
#     plt.bar(i, height=balanced_df[balanced_df.shape[1]-1].value_counts()[i], width=0.5)
# plt.xlabel('Class')
# plt.ylabel("Number of Instances")
# plt.show()


# print('Balanced Class Distribution: ')
# print(balanced_df[balanced_df.shape[1]-1].value_counts())

# print('Balanced Class Distrobution DF shape: ', balanced_df.shape)

# # Finally, separate the features and the labels into X and y variables.
# y = balanced_df[balanced_df.shape[1]-1]
# X = balanced_df.drop(balanced_df.shape[1]-1, axis=1)