import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import datetime

# 创建保存目录
save_dir = "training_results"
os.makedirs(save_dir, exist_ok=True)

# 数据PATH
train_dir = 'Plantvillage/train'
val_dir = 'Plantvillage/valid'
test_dir = 'Plantvillage/test'

def densenet_preprocess(x):
    return tf.keras.applications.densenet.preprocess_input(x)

# 数据增强
train_datagen = ImageDataGenerator(
    preprocessing_function=densenet_preprocess,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=densenet_preprocess)
test_datagen = ImageDataGenerator(preprocessing_function=densenet_preprocess)

# 数据加载
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# 计算训练集类别权重
classes = train_generator.classes
class_weights = compute_class_weight('balanced', 
                                    classes=np.unique(classes),
                                    y=classes)
class_weights = dict(enumerate(class_weights))

# 构建DenseNet121模型
base_model = DenseNet121(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling=None 
)

base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  
    Dense(1024, activation='relu'), 
    Dense(train_generator.num_classes, activation='softmax')
])

# 回调函数
checkpoint_path = os.path.join(save_dir, "best_model_DenseNet121_Plantvillage_10+100")
checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1,
    save_format='tf' 
)

# 编译参数
model.compile(
    optimizer=Adam(learning_rate=0.001),  
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 第一阶段训练
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[checkpoint]
)

# 解冻策略
base_model.trainable = True
for layer in base_model.layers[:-10]:  
    layer.trainable = False
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True  # 保持BN层可训练

# 微调阶段
model.compile(
    optimizer=Adam(learning_rate=1e-4),  
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    epochs=100, 
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[checkpoint]
)

# 模型保存
final_model_path = os.path.join(
    save_dir,
    f"final_DenseNet121_Plantvillage_10+100_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
)
save_model(model, final_model_path, save_format='tf')

# 可视化训练过程
def plot_and_save_history(history, history_fine, save_dir):
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.plot([10-1, 10], plt.ylim(), label='Fine Tuning Start')
    plt.legend(loc='lower right')
    plt.title('Training Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.plot([10-1, 10], plt.ylim(), label='Fine Tuning Start')
    plt.legend(loc='upper right')
    plt.title('Training Loss')

    plt.savefig(os.path.join(save_dir, 'training_curves_DenseNet121_Plantvillage_10+100.png'))
    plt.close()

plot_and_save_history(history, history_fine, save_dir)

# 评估与报告
def save_evaluation_results():
    best_model = tf.keras.models.load_model(checkpoint_path)
    
    test_loss, test_acc = best_model.evaluate(test_generator)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    y_pred = np.argmax(best_model.predict(test_generator), axis=1)
    y_true = test_generator.classes
    
    cm = confusion_matrix(y_true, y_pred)
    
    # 获取类别标签
    class_names = list(test_generator.class_indices.keys())
    
    # 优化绘图参数
    plt.figure(figsize=(12, 10))  # 增大画布尺寸
    ax = sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={'size': 8}  # 减小注释字体
    )
    
    # 旋转并对齐标签
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,  # 45度倾斜
        ha='right',     # 水平对齐方式
        fontsize=8     # 适当减小字体
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,    # 保持水平
        ha='right',     # 对齐方式
        fontsize=8
    )
    
    # 调整边距
    plt.subplots_adjust(
        left=0.3,            # 左侧留出更多空间
        bottom=0.4     # 底部留出更多空间
    )
    
    plt.xlabel('Predicted', fontsize=10)
    plt.ylabel('True', fontsize=10)
    plt.title('Confusion Matrix', fontsize=12)
    
    # 保存图片
    plt.savefig(
        os.path.join(save_dir, 'confusion_matrix_DenseNet121_Plantvillage_10+100.png'),
        dpi=300,                      # 提高分辨率
        bbox_inches='tight'  # 自动裁剪空白
    )
    plt.close()
    
    report = classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys(), digits=4) 
    with open(os.path.join(save_dir, 'classification_report_DenseNet121_Plantvillage_10+100.txt'), 'w') as f:
        f.write(report)
        
    # 保存混淆矩阵数值
    np.savetxt(os.path.join(save_dir, 'confusion_matrix_DenseNet121_Plantvillage_10+100.csv'), cm, fmt='%d', delimiter=',')
    
save_evaluation_results()

print("所有训练结果已保存在目录:", os.path.abspath(save_dir))
