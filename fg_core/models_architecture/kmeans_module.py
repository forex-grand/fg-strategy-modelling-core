import tensorflow as tf
import keras


@keras.saving.register_keras_serializable()
class KMeansClustering(tf.keras.layers.Layer):
    def __init__(self, k, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def get_config(self):
        config = super().get_config()
        config.update({'k': self.k})
        return config

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def kmeans_plus_plus_init(self, X):
        """
        Vectorised D² (k-means++) centroid seeding.

        All distance computations are batched over n_samples so there is
        no Python-level loop over the data.  The only Python loop is over
        k (the number of centroids), which is typically tiny.
        """
        k = self.k
        n_samples = tf.shape(X)[0]

        # Pick the first centroid uniformly at random
        first_idx = tf.random.uniform([], 0, n_samples, dtype=tf.int32)
        chosen = tf.expand_dims(X[first_idx], 0)          # (1, features)

        centroid_indices = tf.TensorArray(dtype=tf.int32, size=k)
        centroid_indices = centroid_indices.write(0, first_idx)

        for i in range(1, k):
            # X          : (n, f)
            # chosen     : (i, f)
            # diff       : (n, i, f)
            diff = tf.expand_dims(X, 1) - tf.expand_dims(chosen, 0)
            sq_dists = tf.reduce_sum(tf.square(diff), axis=2)   # (n, i)
            min_dists = tf.reduce_min(sq_dists, axis=1)         # (n,)  D²

            # Sample next centroid proportional to D²
            probs = min_dists / tf.reduce_sum(min_dists)        # (n,)
            log_probs = tf.math.log(tf.expand_dims(probs, 0))   # (1, n)
            next_idx = tf.cast(
                tf.squeeze(tf.random.categorical(log_probs, 1)),
                tf.int32
            )

            centroid_indices = centroid_indices.write(i, next_idx)
            chosen = tf.concat(
                [chosen, tf.expand_dims(X[next_idx], 0)], axis=0
            )

        return tf.gather(X, centroid_indices.stack())           # (k, f)

    @tf.function
    def re_initialize_centroids_random(self, X):
        """Return k randomly-chosen rows from X (no assignment to self)."""
        n_samples = tf.shape(X)[0]
        shuffled = tf.random.shuffle(tf.range(n_samples))
        random_indices = shuffled[:self.k]
        return tf.cast(tf.gather(X, random_indices), tf.float32)

    # ------------------------------------------------------------------
    # Layer build / call
    # ------------------------------------------------------------------

    def build(self, input_shape):
        self.centroids = self.add_weight(
            name='centroids',
            shape=(self.k, input_shape[-1]),
            initializer='random_uniform',
            trainable=True,
        )
        super().build(input_shape)

    def call(self, X, training=False):
        clusters = self.assign_clusters(X)
        if training:
            loss = self.compute_loss(X, clusters)
            self.add_loss(loss)
            return loss
        return tf.cast(clusters, tf.int32)

    # ------------------------------------------------------------------
    # Core k-means operations
    # ------------------------------------------------------------------

    @tf.function
    def compute_distances(self, X):
        """Return (n_samples, k) squared-distance matrix."""
        X_exp = tf.expand_dims(X, 1)                    # (n, 1, f)
        C_exp = tf.expand_dims(self.centroids, 0)       # (1, k, f)
        return tf.reduce_sum(tf.square(X_exp - C_exp), axis=2)  # (n, k)

    @tf.function
    def assign_clusters(self, X):
        distances = self.compute_distances(X)
        return tf.cast(tf.argmin(distances, axis=1), tf.int32)

    @tf.function
    def compute_loss(self, X, clusters):
        """Mean intra-cluster squared distance (standard k-means objective)."""
        X = tf.cast(X, tf.float32)
        cluster_centers = tf.gather(self.centroids, clusters)   # (n, f)
        distances = tf.reduce_sum(tf.square(X - cluster_centers), axis=1)
        return tf.reduce_mean(distances)

    @tf.function
    def update_centroids(self, X, cluster_assignments):
        """
        Recompute each centroid as the mean of its assigned points.
        Empty clusters keep their current centroid.
        """
        centroids = tf.TensorArray(dtype=tf.float32, size=self.k)
        for k in tf.range(self.k):
            mask = tf.equal(cluster_assignments, k)             # (n,)
            cluster_points = tf.boolean_mask(X, mask)          # (m, f)
            if tf.shape(cluster_points)[0] > 0:
                centroid = tf.reduce_mean(cluster_points, axis=0)
            else:
                # Keep existing centroid for empty clusters
                centroid = self.centroids[k]
            centroids = centroids.write(k, centroid)
        return centroids.stack()                                # (k, f)

    # ------------------------------------------------------------------
    # Training strategies
    # ------------------------------------------------------------------

    def fit_randomizer(self, X, max_iters):
        """
        Random-restart Lloyd's algorithm.
        Reinitialise centroids randomly, run Lloyd's, keep the best result.
        """
        X = tf.cast(X, tf.float32)

        # Baseline: evaluate current centroids
        best_centroids = tf.identity(self.centroids)
        best_loss = self.compute_loss(X, self.assign_clusters(X))

        # Reinitialise and run Lloyd's
        self.centroids.assign(self.re_initialize_centroids_random(X))

        for _ in tf.range(max_iters):
            prev_centroids = tf.identity(self.centroids)
            assignments = self.assign_clusters(X)
            new_centroids = self.update_centroids(X, assignments)
            self.centroids.assign(new_centroids)
            if tf.reduce_all(tf.equal(prev_centroids, new_centroids)):
                break

        loss = self.compute_loss(X, self.assign_clusters(X))

        # Keep whichever run was better
        if loss < best_loss:
            best_loss = loss
            best_centroids = tf.identity(self.centroids)

        self.centroids.assign(best_centroids)
        return best_loss

    def fit_k_plus_plus(self, X, max_iters, verbose=False):
        """
        k-means++ initialisation followed by Lloyd's algorithm.
        Uses vectorised D² seeding — no Python loop over n_samples.
        """
        X = tf.cast(X, tf.float32)

        if len(X.shape) > 2:
            X = tf.squeeze(X, axis=0)

        # Seed centroids with k-means++
        self.centroids.assign(self.kmeans_plus_plus_init(X))

        for step in tf.range(max_iters):
            prev_centroids = tf.identity(self.centroids)

            # Use assign_clusters directly — NOT self() — to avoid the
            # training branch and spurious add_loss calls.
            assignments = self.assign_clusters(X)
            new_centroids = self.update_centroids(X, assignments)
            self.centroids.assign(new_centroids)

            if verbose:
                loss = self.compute_loss(X, assignments)
                tf.print(f"iter", step, "loss:", loss)

            if tf.reduce_all(tf.equal(prev_centroids, new_centroids)):
                break

        final_assignments = self.assign_clusters(X)
        return self.compute_loss(X, final_assignments)


# ---------------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class Kmeans_model(tf.keras.Model):
    def __init__(self, k, max_iters=100, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.max_iters = max_iters
        self.kmeans_layer = KMeansClustering(k=k)

    def get_config(self):
        config = super().get_config()
        config.update({'k': self.k, 'max_iters': self.max_iters})
        return config

    @classmethod
    def from_config(cls, config):
        # Strip Keras-managed keys that the constructor doesn't accept
        for key in ('name', 'trainable', 'dtype'):
            config.pop(key, None)
        return cls(**config)

    def build(self, input_shape):
        if self.built:
            return
        n_samples = input_shape[0]
        if n_samples is not None and self.k > n_samples:
            raise ValueError(
                f"k ({self.k}) cannot be greater than n_samples ({n_samples})."
            )
        if not isinstance(input_shape, tf.TensorShape):
            input_shape = tf.TensorShape(input_shape)
        # KMeansClustering.build expects (k, n_features) but internally only
        # uses input_shape[-1], so pass the full data shape directly.
        self.kmeans_layer.build(input_shape)
        super().build(input_shape)

    def compile(self, training_strategy: str = "randomizer", **kwargs):
        """training_strategy: 'randomizer' | 'kplus'"""
        super().compile(**kwargs)
        self.training_strategy = training_strategy

    def call(self, inputs):
        return self.kmeans_layer(inputs)

    def fit(self, X, epochs: int = 1, verbose: bool = False, **kwargs):
        # Materialise a tf.data.Dataset into a single tensor
        if isinstance(X, tf.data.Dataset):
            X = tf.cast(
                tf.concat(list(X.as_numpy_iterator()), axis=0), tf.float32
            )
        else:
            X = tf.cast(X, tf.float32)

        print(f"Data shape: {X.shape}")
        self.build(X.shape)

        best_loss = float('inf')
        for epoch in range(epochs):
            if self.training_strategy == "randomizer":
                loss = self.kmeans_layer.fit_randomizer(X, self.max_iters)
            elif self.training_strategy == "kplus":
                loss = self.kmeans_layer.fit_k_plus_plus(
                    X, self.max_iters, verbose=verbose
                )
            else:
                raise ValueError(
                    f"Unknown training_strategy '{self.training_strategy}'. "
                    "Choose 'randomizer' or 'kplus'."
                )

            loss_val = float(loss)
            print(f"Epoch {epoch + 1}/{epochs} — loss: {loss_val:.6f}")
            if loss_val < best_loss:
                best_loss = loss_val

        print(f"Best loss: {best_loss:.6f}")
        return best_loss

    def predict(self, X):
        """Return cluster assignments for X."""
        X = tf.cast(X, tf.float32)
        return self.kmeans_layer.assign_clusters(X).numpy()

    @property
    def cluster_centers_(self):
        """Convenience accessor for the trained centroids."""
        return self.kmeans_layer.centroids.numpy()
