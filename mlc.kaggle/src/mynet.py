import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50 
from keras.backend import permute_dimensions
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Dropout, Permute
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import fbeta_score

class MyNet:
    def __init__(self, net_selection="vgg16", img_dim=(3, 272, 480), num_classes=17):

        # Pretrained model with imagenet weights
        if net_selection=="resnet50":
            base_model = ResNet50(include_top=False,
                               weights='imagenet',
                               input_shape=img_dim)
            last_layer = 173
        else:
            base_model = VGG16(include_top=False,
                               weights='imagenet',
                               input_shape=img_dim)
            last_layer = 18

        # Attaching layers to tail of base_model 
        x = Flatten()(base_model.layers[last_layer].output)
        output = Dense(num_classes, activation='sigmoid')(x)
        # Reshape model output if necessary
        # output = Reshape((17, 1, 1))(output)

        stacked_model = Model(base_model.input, output)
        self.model = stacked_model

    def predict(self, preprocessor, mode=0, batch_size=32):
        """
        Launch the predictions on the test dataset as well as the additional test dataset
        mode: 
            0 - test(default)
            1 - train
            2 - validation
        :return:
            predictions: list
                An array containing num_classes length long arrays of raw prediction value.
            filenames: list
                File names associated to each prediction
        """
        # IMPORTANT:: shuffle=False in evaluation. We do not need shuffle in prediction!!!
        # Caused problems when comparing to y_test due to shuffling.
        # See this post: https://github.com/keras-team/keras/issues/3477
        if mode == 0:
            generator = preprocessor.get_prediction_generator(batch_size, shuffle=False)
            X = preprocessor.X_test
        elif mode == 1:
            generator = preprocessor.get_train_generator(batch_size, shuffle=False)
            X = preprocessor.X_train
        elif mode == 2:
            generator = preprocessor.get_val_generator(batch_size, shuffle=False)
            X = preprocessor.X_val
        else:
            AssertionError ("Prediction mode not supported!")
            return

        predictions = self.model.predict_generator(generator=generator, verbose=1,
                                                     steps=len(X) / batch_size)
        assert len(predictions) == len(X), \
            "len(predictions) = {}, len(X) = {}".format(
                len(predictions), len(X))
        return predictions, np.array(X)

    def map_predictions(self, preprocessor, predictions, thresholds):
            """
            Return the predictions mapped to their labels
            :param predictions: the predictions from the predict() method
            :param thresholds: The threshold of each class to be considered as existing or not existing
            :return: the predictions list mapped to their labels
            """
            predictions_labels = []
            for prediction in predictions:
                labels = [preprocessor.y_map[i] for i, value in enumerate(prediction) if value > thresholds[i]]
                predictions_labels.append(labels)

            return predictions_labels

    def fbeta(self, preprocessor, mode=0, THRESHOLD=0.2):
        """
        mode: 
            0 - test(default)
            1 - train
            2 - validation
        """
        pred, _ = self.predict(preprocessor, mode)

        if mode == 0:
            y = preprocessor.y_test
        elif mode == 1:
            y = preprocessor.y_train
        elif mode == 2:
            y = preprocessor.y_val
        else:
            AssertionError ("Prediction mode not supported!")
            return

        return fbeta_score(np.array(y), pred > THRESHOLD, beta=2, average='samples')

    def _replace_intermediate_layer_in_keras(self, model, layer_id, new_layer):
    
        layers = [l for l in model.layers]
    
        x = layers[0].output
        for i in range(1, len(layers)):
            if i == layer_id:
                x = new_layer(x)
            else:
                x = layers[i](x)
    
        new_model = Model(inputs=layers[0].input, outputs=x)
        return new_model
    
    def _insert_intermediate_layer_in_keras(self, model, layer_id, new_layer):
    
        layers = [l for l in model.layers]
    
        x = layers[0].output
        for i in range(1, len(layers)):
            if i == layer_id:
                x = new_layer(x)
            x = layers[i](x)
    
        new_model = Model(inputs=layers[0].input, outputs=x)
        return new_model
