import os
import tensorflow as tf
import numpy as np
import keras

@keras.saving.register_keras_serializable()
class KMeansClustering(tf.keras.layers.Layer):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.centroids = None

    def kmeans_plus_plus_init(self, X, k):
        n_samples = tf.shape(X)[0]
        centroids = []
        
        # Select first centroid randomly
        first_centroid_idx = tf.random.uniform([], 0, n_samples, dtype=tf.int32)
        centroids.append(X[first_centroid_idx])
        
        # Select remaining centroids
        for _ in tf.range(1, k):
            # Calculate distances from points to the centroids
            distances = tf.stack([
                tf.reduce_min([
                    tf.reduce_sum(tf.square(X[i] - c)) for c in centroids
                ]) for i in range(n_samples)
            ])
            
            # Calculate probabilities for each point
            probabilities = tf.square(distances) / tf.reduce_sum(tf.square(distances))
            probabilities = tf.reshape(probabilities, [1, -1])  # Reshape for categorical
            
            # Select next centroid
            next_centroid_idx = tf.random.categorical(tf.math.log(probabilities), 1)[0][0]
            centroids.append(X[next_centroid_idx])
        
        return tf.stack(centroids)

    def build(self, input_shape):
        if self.centroids is None:
            num_features = input_shape[-1]
            self.centroids = self.add_weight(
            name='centroids',
            shape=(self.k, input_shape[-1]),
            initializer='random_uniform',
            trainable=True
        )
        super().build(input_shape)

    def initialize_centroids(self, X):
        n_samples = tf.shape(X)[0]
        shuffled_samples = tf.random.shuffle(tf.range(n_samples), seed=self.seed)
        random_indices = tf.slice(shuffled_samples, [0], [self.k])
        self.centroids = tf.Variable(tf.cast(tf.gather(X, random_indices), dtype=tf.float32), trainable=True)
        return self.centroids

    @tf.function
    def re_initialize_centroids(self, X):
        n_samples = tf.shape(X)[0]
        
        indices = tf.range(n_samples)
        shuffled_indices = tf.random.shuffle(indices)
        
        random_indices = tf.slice(shuffled_indices, [0], [self.k])
        
        centroids = tf.gather(X, random_indices)
        centroids = tf.cast(centroids, dtype=tf.float32)
        
        return centroids

    @tf.function
    def call(self, X, training=False):
        clusters = self.assign_clusters(X)
        if training:
            loss = self.fit_randomizer(X, 10)
            print("training: ", loss)
            self.add_loss(loss)
            return loss

        return tf.cast(clusters, dtype=tf.int32)

    @tf.function
    def compute_distances(self, X):
        X_expanded = tf.expand_dims(X, 1)
        centroids_expanded = tf.expand_dims(self.centroids, 0)
        distances = tf.reduce_sum(tf.square(X_expanded - centroids_expanded), axis=2)
        return distances

    @tf.function
    def assign_clusters(self, X):
        distances = self.compute_distances(X)
        cluster_assignments = tf.argmin(distances, axis=1)
        return tf.cast(cluster_assignments, dtype=tf.int32)

    @tf.function
    def compute_loss(self, X, clusters):
        """
          Returns mean Squared error.
        """
        X = tf.cast(X, dtype=tf.float32)
        cluster_centers = tf.gather(self.centroids, clusters)
        distances = tf.reduce_sum(tf.square(X - cluster_centers), axis=1)
        return tf.reduce_mean(distances)

    @tf.function
    def update_centroids(self, X, cluster_assignments):
        centroids = tf.TensorArray(dtype=tf.float32, size=self.k)
        for k in tf.range(self.k):
            cluster_points = tf.gather(X, tf.where(tf.equal(cluster_assignments, k))[:, 0])

            if tf.size(cluster_points) > 0:
                centroid = tf.reduce_mean(cluster_points, axis=0)
            else:
                centroid = X[tf.random.uniform([], 0, tf.shape(X)[0], dtype=tf.int32)]

            centroids = centroids.write(k, centroid)

        return centroids.stack()

    def fit_randomizer(self, X, max_iters, **kwargs):
        X = tf.cast(X, dtype=tf.float32)

        best_loss_centroids = tf.identity(self.centroids)
        best_loss_value = self.compute_loss(X, self.assign_clusters(X))

        self.centroids.assign(self.re_initialize_centroids(X))

        cluster_assignments = self.assign_clusters(X)

        for _ in tf.range(max_iters):
            prev_centroids = self.centroids

            cluster_assignments = self.assign_clusters(X)

            centroids = self.update_centroids(X, cluster_assignments)

            if tf.reduce_all(tf.equal(prev_centroids, centroids)):
                break

            self.centroids.assign(centroids)

        centroids_ =  self.assign_clusters(X)

        loss = self.compute_loss(X, centroids_)

        if loss<best_loss_value:
            best_loss_value = loss
            best_loss_centroids = tf.identity(self.centroids)

        self.centroids.assign(best_loss_centroids)
        centroids_ =  self.assign_clusters(X)
        return loss

    def fit_k_plus_plus(self, X, max_iters, **kwargs):
        X = tf.cast(X, dtype=tf.float32)

        if len(X.shape)>2:
          X = tf.squeeze(X, axis=0)


        if self.centroids is None:
          self.initialize_centroids(X)
          self.centroids.assign(self.kmeans_plus_plus_init(X, self.k))
        else:
          self.centroids.assign(self.kmeans_plus_plus_init(X, self.k))

        for _ in tf.range(max_iters):
            prev_centroids = self.centroids

            cluster_assignments = self(X)

            centroids = self.update_centroids(X, cluster_assignments)

            if tf.reduce_all(tf.equal(prev_centroids, centroids)):
                break

            self.centroids.assign(centroids)
            if kwargs.get("verbose"):
                loss = self.compute_loss(X, cluster_assignments)
                print(f"iter {_} loss: {loss}")

        centroids_ =  self.assign_clusters(X)
        loss = self.compute_loss(X, centroids_).numpy()
        return loss

@keras.saving.register_keras_serializable()
class Kmeans_model(tf.keras.Model):
    def __init__(self, k, max_iters=100, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.max_iters = max_iters
        self.kmeans_layer = KMeansClustering(k=k)

    def get_config(self):
        config = super().get_config()
        config.update({
            'k': self.k,
            'max_iters': self.max_iters
        })
        return config

    @classmethod
    def from_config(cls, config):
        config.pop('name', None)
        config.pop('trainable', None)
        config.pop('dtype', None)

        return cls(**config)

    def build(self, input_shape):
        if self.k>input_shape[0]:
            raise Exception("Neighbor cannot be greater than sample size.")
             
        if not self.built:
            if not isinstance(input_shape, tf.TensorShape):
                input_shape = tf.TensorShape(input_shape)

            classes_shape = (self.k, input_shape[1])

            self.kmeans_layer.build(classes_shape)
            super().build(input_shape)
            self.built = True

    def compile(self, training_strategy: str="randomizer", **kwargs):
        """"training strategies: randomizer, kplus"""
        super().compile(**kwargs)
        self.training_strategy = training_strategy

    def call(self, inputs):
        return self.kmeans_layer(inputs)

    def fit(self, X, epochs: int=100, **kwargs):
        
        if isinstance(X, tf.data.Dataset):
            # Collect all data from the dataset
            X = tf.concat(list(X.as_numpy_iterator()), axis=0)
            X = tf.cast(X, dtype=tf.float32)
        
        print(X.shape)
        self.build(X.shape)
        
        for epoch in tf.range(epochs):
            if self.training_strategy == "randomizer":
                loss = self.kmeans_layer.fit_randomizer(X, self.max_iters, **kwargs)
                print(f"epoch {epoch} Mean squared loss: ", loss)
            elif self.training_strategy=='kplus':
                loss = self.kmeans_layer.fit_k_plus_plus(X, self.max_iters)
                print(f"epoch {epoch} Mean squared loss: ", loss)

        centroids_ = self.kmeans_layer.assign_clusters(X)
        loss = self.kmeans_layer.compute_loss(X, centroids_).numpy()
        print("Best loss: ", loss)
        return loss
