<?php
// 中間層のニューロン数
define('HIDDEN_NEURONS', [12, 8]);

// 中間層の活性化関数 (sigmoid, relu, tanhなど)
define("HIDDEN_NEURONS_ACTIVATION_FUNCS", [
    "relu", // 第1中間層
    "relu", // 第2中間層
]);

// 出力層の活性化関数
define('OUTPUT_ACTIVATION_FUNC', 'softmax');

// 損失関数 (mse または cross_entropy)
define('LOSS_FUNCTION', 'cross_entropy');

define('SAVE_INTERVAL_EPOCHS', 5000); // モデルの定期保存の間隔 (エポック数)
define('ENABLE_SAVE_INTERVAL_PREFIX_EPOCH', false); // モデルの定期保存のファイル名にエポック数を付与する

define('STOP_PCNTL_SIGNAL', false); // Ctrl+Cでの保存確認


define('MODEL_FILE', 'model.json'); // モデルの保存ファイル名
define('VOCABULARY_FILE', 'vocabulary.json'); // 語彙の保存ファイル名


//_/_/_/_/_/_/_/_/_/_/_/_/_/_/
// 学習用

define('EPOCHS', 10000); // 繰り返し数
define('INIT_LEARNING_RATE', 0.1); // 初期学習率
define('DECAY_RATE', 0.001); // 学習率の減衰率 (0であれば一定)

// ラベル
define('LABELS', ['positive', 'negative', 'neutral']);

// 教師データ
define('CORPUS', [
    ['text' => 'i love this amazing place', 'label' => 'positive'],
    ['text' => 'this is great and wonderful', 'label' => 'positive'],
    ['text' => 'excellent work good job', 'label' => 'positive'],
    ['text' => 'i hate this awful place', 'label' => 'negative'],
    ['text' => 'this is bad and terrible', 'label' => 'negative'],
    ['text' => 'poor work bad job', 'label' => 'negative'],
    ['text' => 'this is a place', 'label' => 'neutral'],
    ['text' => 'the work is done', 'label' => 'neutral'],
    ['text' => 'i see the job', 'label' => 'neutral'],
]);

//_/_/_/_/_/_/_/_/_/_/_/_/_/_/
// 推論用

// 入力データ
define('TEST_SENTENCES', [
    "this is a good place",
    "i hate bad work",
    "this job is done",
    "amazing wonderful great"
]);