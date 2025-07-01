<?php
require_once './Config.php';

require_once './src/PHPMLearning/ActivationFunctions.php';
require_once './src/PHPMLearning/NeuralNetwork.php';

use PHPMLearning\NeuralNetwork;

function main(int $argc, array $argv) : int {
    // モデルと語彙データの読み込み
    if (!file_exists(MODEL_FILE) || !file_exists(VOCABULARY_FILE)) {
        printf("Error: %s or %s not found. Please run train.php first.\n", MODEL_FILE, VOCABULARY_FILE);
        return 1;
    }

    $model_data = json_decode(file_get_contents(MODEL_FILE), true);
    $vocab_data = json_decode(file_get_contents(VOCABULARY_FILE), true);
    $vocabulary = $vocab_data['vocabulary'];
    $labels = $vocab_data['labels'];
    $size = count($vocabulary);

    printf("Loaded %s, %s\n", MODEL_FILE, VOCABULARY_FILE);

    // ニューラルネットワークの構築
    $nn = new NeuralNetwork();
    $nn->loadModel($model_data);

    foreach (TEST_SENTENCES as $sentence) {
        // テキストをベクトル化
        $features = array_fill(0, $size, 0.0);
        $words = explode(' ', $sentence);
        foreach ($words as $word) {
            $index = array_search($word, $vocabulary);
            if ($index !== false) {
                $features[$index] = 1.0;
            }
        }
        
        // 推論
        $output = $nn->predict($features);
        $predicted_label = $labels[$output['label']];
        
        printf("\nInput: \"%s\"\n", $sentence);
        printf(" -> Label: %s\n", $predicted_label);
        printf("    Probabilities:\n");
        foreach ($output['probs'] as $index => $prob) {
            printf("      - %s: %.2f%%\n", $labels[$index], $prob * 100);
        }
    }

    return 0;
}

exit(main($argc, $argv));
