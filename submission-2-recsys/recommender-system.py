import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ratings_df = pd.read_csv("animelist.csv")
anime_df = pd.read_csv("anime.csv")

all_ratings = ratings_df[["anime_id", "user_id", "rating"]]
all_anime = anime_df[["MAL_ID", "Name", "Genres", "Type", "Studios", "Source"]]


# Content-Based Filtering
def process_multilabel(series):
    series = series.split(", ")
    if "Unknown" in series:
        series.remove("Unknown")
    return ", ".join(series)


all_anime.loc[:, "Genres"] = all_anime["Genres"].map(process_multilabel)
all_anime.loc[:, "Source"] = all_anime["Source"].map(process_multilabel)
all_anime.loc[:, "Type"] = all_anime["Type"].map(process_multilabel)

all_anime.loc[:, "features"] = (
    all_anime["Genres"] + " " + all_anime["Type"] + " " + all_anime["Source"]
)

tfidf = TfidfVectorizer()

tfidf.fit(all_anime["features"])

tfidf_matrix = tfidf.fit_transform(all_anime["features"])
cosine_sim = cosine_similarity(tfidf_matrix)

cosine_sim_df = pd.DataFrame(
    cosine_sim, index=all_anime["Name"], columns=all_anime["Name"]
)


def anime_recommendation(
    nama_anime: str,
    similarity_data=cosine_sim_df,
    items=all_anime[["Name", "Genres", "Studios", "Type", "Source"]],
    k=10,
):
    """
    Rekomendasi Anime berdasarkan kemiripan dataframe

    Parameter:
    ---
    nama_anime : tipe data string (str)
                Nama Anime (index kemiripan dataframe)
    similarity_data : tipe data pd.DataFrame (object)
                      Kesamaan dataframe, simetrik, dengan anime sebagai
                      indeks dan kolom
    items : tipe data pd.DataFrame (object)
            Mengandung kedua nama dan fitur lainnya yang digunakan untuk mendefinisikan kemiripan
    k : tipe data integer (int)
        Banyaknya jumlah rekomendasi yang diberikan
    ---


    Pada index ini, kita mengambil k dengan nilai similarity terbesar
    pada index matrix yang diberikan (i).
    """

    # Mengambil data dengan menggunakan argpartition untuk melakukan partisi secara tidak langsung sepanjang sumbu yang diberikan
    # Dataframe diubah menjadi numpy
    # Range(start, stop, step)
    index = (
        similarity_data.loc[:, nama_anime].to_numpy().argpartition(range(-1, -k, -1))
    )

    # Mengambil data dengan similarity terbesar dari index yang ada
    closest = similarity_data.columns[index[-1 : -(k + 2) : -1]]

    # Drop nama_resto agar nama resto yang dicari tidak muncul dalam daftar rekomendasi
    closest = closest.drop(nama_anime, errors="ignore")

    print("=========" * 10)
    print(f"Top {k} anime yang mirip dengan: {nama_anime}")
    print("=========" * 10)

    return pd.DataFrame(closest).merge(items).head(k)


print(anime_recommendation("Dragon Ball Z"))

# Collaborative Filtering

all_ratings = all_ratings.sample(100_000, random_state=42).reset_index(drop=True)

user_ids = all_ratings["user_id"].unique().tolist()
anime_ids = all_ratings["anime_id"].unique().tolist()

# Encoding user id
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
user_encoded2user = {i: x for i, x in enumerate(user_ids)}

# Encoding anime id
anime2anime_encoded = {x: i for i, x in enumerate(anime_ids)}
anime_encoded2anime = {i: x for i, x in enumerate(anime_ids)}

all_ratings["user"] = all_ratings["user_id"].map(user2user_encoded)
all_ratings["anime"] = all_ratings["anime_id"].map(anime2anime_encoded)

min_rating = np.min(all_ratings["rating"])
max_rating = np.max(all_ratings["rating"])

# Shuffle
all_ratings = all_ratings.sample(frac=1, random_state=42)

X = all_ratings[["user", "anime"]].values
y = (
    all_ratings["rating"]
    .apply(lambda x: (x - min_rating) / (max_rating - min_rating))
    .values
)

# Split
train_indices = int(0.95 * all_ratings.shape[0])

X_train, X_test, y_train, y_test = (
    X[:train_indices],
    X[train_indices:],
    y[:train_indices],
    y[train_indices:],
)

X_train_array = [X_train[:, 0], X_train[:, 1]]
X_test_array = [X_test[:, 0], X_test[:, 1]]


def RecommenderNet(n_users, n_animes, embedding_size=128):
    user = tf.keras.layers.Input(name="user", shape=[1])
    user_embedding = tf.keras.layers.Embedding(
        name="user_embedding", input_dim=n_users, output_dim=embedding_size
    )(user)
    user_bias = tf.keras.layers.Embedding(
        name="user_bias", input_dim=n_users, output_dim=embedding_size
    )(user)

    anime = tf.keras.layers.Input(name="anime", shape=[1])
    anime_embedding = tf.keras.layers.Embedding(
        name="anime_embedding", input_dim=n_animes, output_dim=embedding_size
    )(anime)
    anime_bias = tf.keras.layers.Embedding(
        name="anime_bias", input_dim=n_animes, output_dim=embedding_size
    )(anime)

    x = tf.keras.layers.Dot(name="dot_product", normalize=True, axes=2)(
        [user_embedding, anime_embedding]
    )
    x = tf.keras.layers.Add()([x, user_bias, anime_bias])
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(1, kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    model = tf.keras.models.Model(inputs=[user, anime], outputs=x)
    model.compile(loss="binary_crossentropy", metrics=["mae", "mse"], optimizer="adam")

    return model


model = RecommenderNet(len(user2user_encoded), len(anime2anime_encoded))

print(model.summary())


filepath = "Checkpoints/anime-cf.h5"
model_checkpoints = tf.keras.callbacks.ModelCheckpoint(
    filepath=filepath,
    save_weights_only=True,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=3, monitor="val_loss", mode="min", restore_best_weights=True
)

my_callbacks = [
    model_checkpoints,
    early_stopping,
]

history = model.fit(
    x=X_train_array,
    y=y_train,
    batch_size=10000,
    epochs=20,
    verbose=1,
    validation_data=(X_test_array, y_test),
    callbacks=my_callbacks,
)

model.load_weights(filepath)

random_user = np.random.choice(user_ids, 1)[0]
print("Random user id: ", random_user)

anime_watched_by_user = all_ratings[all_ratings["user_id"] == random_user]

anime_not_watched = anime_df[
    ~anime_df["MAL_ID"].isin(anime_watched_by_user["anime_id"].values)
]["MAL_ID"]
anime_not_watched = list(
    set(anime_not_watched).intersection(set(anime2anime_encoded.keys()))
)

anime_not_watched = [[anime2anime_encoded.get(x)] for x in anime_not_watched]
user_encoder = user2user_encoded.get(random_user)
user_anime_array = np.hstack(
    ([[user_encoder]] * len(anime_not_watched), anime_not_watched)
)

user_anime_array = [user_anime_array[:, 0], user_anime_array[:, 1]]

ratings = model.predict(user_anime_array).flatten()

top_ratings_indices = ratings.argsort()[-20:][::-1]
recommended_anime_ids = [
    anime2anime_encoded.get(anime_not_watched[x][0]) for x in top_ratings_indices
]

print("Showing recommendations for users: {}".format(random_user))
print("===" * 9)
print("Anime with high ratings from user")
print("----" * 8)

top_anime_user = (
    anime_watched_by_user.sort_values(by="rating", ascending=False)
    .head(10)["anime_id"]
    .values
)
anime_df_rows = all_anime[all_anime["MAL_ID"].isin(top_anime_user)]
for row in anime_df_rows.itertuples():
    print(row.Name, ":", row.Genres)

print("----" * 8)
print("Top 10 anime recommendation")
print("----" * 8)

recommended_anime = all_anime[all_anime["MAL_ID"].isin(recommended_anime_ids)]
print(recommended_anime)
