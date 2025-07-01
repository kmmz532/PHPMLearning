<?php
require_once './Config.php';

require_once './src/PHPMLearning/ActivationFunctions.php';
require_once './src/PHPMLearning/NeuralNetwork.php';

use PHPMLearning\NeuralNetwork;

function main(int $argc, array $argv) : int {
    // 語彙データを作成する
    $vocabulary = [];
    foreach (CORPUS as $doc) {
        $words = explode(' ', $doc['text']);
        foreach ($words as $word) {
            if (!in_array($word, $vocabulary)) {
                $vocabulary[] = $word;
            }
        }
    }

    sort($vocabulary);
    $size = count($vocabulary);
    printf("Vocabulary size: %s\n", $size);

    // 語彙データの保存
    file_put_contents(VOCABULARY_FILE, json_encode(['vocabulary' => $vocabulary, 'labels' => LABELS], JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE));
    printf("Vocabulary has been saved to %s\n", VOCABULARY_FILE);

    // テキストをベクトル化
    $dataset = [];
    foreach (CORPUS as $doc) {
        // 特徴量ベクトル
        $features = array_fill(0, $size, 0.0);
        $words = explode(' ', $doc['text']);
        foreach ($words as $word) {
            $index = array_search($word, $vocabulary);
            if ($index !== false) {
                $features[$index] = 1.0;
            }
        }

        // ラベル (One-Hot)
        $label_index = array_search($doc['label'], LABELS);
        $label_vector = array_fill(0, count(LABELS), 0.0);
        $label_vector[$label_index] = 1.0;
        
        $dataset[] = ['features' => $features, 'label' => $label_vector];
    }

    // ニューラルネットワークの構築
    $input_neurons = $size;
    $hidden_neurons = HIDDEN_NEURONS;
    $output_neurons = count(LABELS);

    $nn = new NeuralNetwork($hidden_neurons, $input_neurons, $output_neurons, LOSS_FUNCTION);

    $hidden_layer_count = count($hidden_neurons);

    // 中間層の活性化関数の設定
    if (defined('HIDDEN_NEURONS_ACTIVATION_FUNCS')) {
        foreach (HIDDEN_NEURONS_ACTIVATION_FUNCS as $i => $func) {
            $nn->setActivationFunction($i, $func);
        }
    } else {
        // デフォルトはReLU
        for ($i = 0; $i < $hidden_layer_count; $i++) {
            $nn->setActivationFunction($i, 'relu');
        }
    }

    // 出力層の活性化関数の設定
    if (defined('OUTPUT_ACTIVATION_FUNC')) {
        $nn->setActivationFunction($hidden_layer_count, OUTPUT_ACTIVATION_FUNC);
    } else {
        // デフォルトはsoftmax
        $nn->setActivationFunction($hidden_layer_count, 'softmax');
    }

    $init_epoch = 0;
    $learning_rate = INIT_LEARNING_RATE;

    // もしファイルが存在すれば、モデルを読み込み、継続して学習
    if (file_exists(MODEL_FILE)) {
        $model_data = json_decode(file_get_contents(MODEL_FILE), true);
        $nn->loadModel($model_data);
        printf("Model loaded from %s\n", MODEL_FILE);

        if (isset($model_data['epoch']))
            $init_epoch = $model_data['epoch'];

        if (isset($model_data['learning_rate']))
            $learning_rate = $model_data['learning_rate'];

        if (isset($model_data['seed'])) {
            $nn->setSeed($model_data['seed']);
        }
    }

    $nn->initWeights();

    // 途中終了
    if (STOP_PCNTL_SIGNAL && function_exists('pcntl_signal')) {
        declare(ticks=1);
        pcntl_signal(SIGINT, function() use (&$nn) {
            echo "\nDo you want to save the model? (y/n): ";
            $handle = fopen("php://stdin","r");
            $line = fgets($handle);
            $line = trim(strtolower($line));
            if ($line === 'y' || $line === 'yes') {
                echo "Saving model...\n";
                $model_data = $nn->getModel();
                file_put_contents(MODEL_FILE, json_encode($model_data, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE));
                echo "Saved to \"" . MODEL_FILE . "\".\n";
            } else {
                echo "Exiting without saving.\n";
            }
            fclose($handle);
            exit(0);
        });
    }
    
    // 学習
    echo "Training for " . EPOCHS . " epochs...\n";

    for ($epoch = $init_epoch; $epoch < EPOCHS; $epoch++) {
        // データセットをシャッフル
        shuffle($dataset);
        $loss = $nn->train($dataset, $learning_rate);
        
        if ($epoch % 200 === 0) {
            printf("Epoch: %d/%d, Loss: %.6f, Learning Rate: %.6f\n", $epoch + 1, EPOCHS, $loss, $learning_rate);
        }

        if (DECAY_RATE > 0)
            $learning_rate = INIT_LEARNING_RATE / (1 + DECAY_RATE * $epoch);

        if ($epoch % SAVE_INTERVAL_EPOCHS === 0 && $epoch > 0) {
            $model_data = $nn->getModel();
            $model_data['epoch'] = $epoch;

            $filename = ENABLE_SAVE_INTERVAL_PREFIX_EPOCH ? basename(MODEL_FILE, '.json') . "_" . $epoch . ".json" : MODEL_FILE;

            file_put_contents($filename, json_encode($model_data, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE));
            printf("Model has been saved to \"%s\" at epoch %d\n", $filename, $epoch);
        }
    }

    echo "\nTraining finished\n";

    // モデルの保存
    $model_data = $nn->getModel();
    $model_data['epoch'] = EPOCHS;
    file_put_contents(MODEL_FILE, json_encode($model_data, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE));

    printf("Model has been saved to %s\n", MODEL_FILE);

    return 0;
}

exit(main($argc, $argv));
