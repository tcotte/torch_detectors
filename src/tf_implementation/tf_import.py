if __name__ == "__main__":
    import tensorflow as tf
    import keras_cv

    print(tf.__version__)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    bbox = tf.ragged.constant([1,1])
    print(bbox)
    print(tf.sysconfig.get_build_info()['cuda_version'])