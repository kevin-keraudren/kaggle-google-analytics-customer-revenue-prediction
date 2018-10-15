from train import *

if __name__ == "__main__":
    user_features = {}
    user_scores = {}
    for f in list(glob("train/*.csv"))[:10]:
        user_features, user_scores = encode_all_users(f, user_features, user_scores)

    user_features, user_scores = split_users(user_features, user_scores, max_visits=10)

    positive_user_features = {}
    positive_user_scores = {}
    for user, score in user_scores.items():
        if np.sum(score) > 0:
            positive_user_features[user] = user_features[user]
            positive_user_scores[user] = score

    user_features = positive_user_features
    user_scores = positive_user_scores

    max_seq_length = 10
    feature_size = len(user_features[list(user_features.keys())[0]][0])
    model = get_model(max_seq_length, feature_size, filename='s2s.h5')

    n = 0
    for user, features in user_features.items():
        seq_len = len(features)
        input_data = np.zeros((1, max_seq_length, feature_size), dtype='float32')
        input_data[0, :len(features)] = features
        prediction = predict(model, input_data, seq_len)
        print(user, prediction, user_scores[user])
        n += 1
        if n == 20:
            exit(0)
