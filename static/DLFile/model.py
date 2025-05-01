from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, Add, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, Multiply
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout, Dense, Flatten, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model

WORKERS = 2
CHANNEL = 3

IMG_SIZE = 224  # 可根据实际模型输入调整
NUM_CLASSES = 5
TRAIN_NUM = 1000 # use 1000 when you just want to explore new idea, use -1 for full train


def se_block(input_tensor, ratio=16):
    channel_axis = -1
    channels = input_tensor.shape[channel_axis]

    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, channels))(se)
    se = Dense(channels // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(channels, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    return Multiply()([input_tensor, se])


def build_GuCNN_module(input_shape):
    inputs = Input(shape=input_shape)

    def residual_block(x, filters, kernel_size=3):
        shortcut = x
        if x.shape[-1] != filters:
            shortcut = Conv2D(filters, (1, 1), padding="same", kernel_regularizer=l2(0.0005))(x)
        x = Conv2D(filters, (kernel_size, kernel_size), padding="same", kernel_regularizer=l2(0.0005))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters, (kernel_size, kernel_size), padding="same", kernel_regularizer=l2(0.0005))(x)
        x = BatchNormalization()(x)

        x = se_block(x)

        x = Add()([x, shortcut])  # Residual connection
        x = ReLU()(x)
        return x

    x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(0.0005))(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = residual_block(x, 64)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = residual_block(x, 128)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), padding="same", kernel_regularizer=l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(256, (3, 3), padding="same", kernel_regularizer=l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = residual_block(x, 256)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(512, (3, 3), padding="same", kernel_regularizer=l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = residual_block(x, 512)

    feature_for_gradcam = x

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # model = Model(inputs, x, name="GuCNN_Module_with_Residual")
    model = Model(inputs, [feature_for_gradcam, x], name="GuCNN_Module_with_Residual")
    return model


# 构建 GuCNN 模块
GuCNN_layer = build_GuCNN_module(input_shape=(IMG_SIZE, IMG_SIZE, 3))
GuCNN_layer.summary()

input_tensor = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
features = GuCNN_layer(input_tensor)



def build_my_model(img_size=224, num_classes=5):
    GuCNN_layer = build_GuCNN_module(input_shape=(img_size, img_size, 3))
    input_tensor = Input(shape=(img_size, img_size, 3))
    features = GuCNN_layer(input_tensor)

    x = GlobalAveragePooling2D()(features[1])
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=input_tensor, outputs=output)
    return model
