<?php
/** Softmax関数スクリプト
 * 
 * コマンドライン上で引数として与えられた数値に対して
 * ソフトマックス関数を適用してその結果を表示する
 * 
 * そして本当に合計すると1になるかを確認する
 * 
 * Usage: php softmax.php 1.0 2.0 3.0
 */

require_once '../PHPMLearning/ActivationFunctions.php';

use PHPMLearning\ActivationFunctions;

/**
 * メイン関数
 * @param int $argc 引数の数
 * @param string[] $argv 引数
 * @return int 終了コード
 */
function main(int $argc, array $argv) : int {
    if ($argc < 2) {
        fprintf(STDERR, "Usage: php %s <arg1> <arg2> ...\n", $argv[0]);
        return 1;
    }

    $inputs = array_map('floatval', array_slice($argv, 1));
    $outputs = ActivationFunctions::softmax($inputs);
    $size = $argc - 1;

    printf("softmax(");
    for ($i = 0; $i < $size; $i++) {
        printf("%.1f", $inputs[$i]);
        if ($i < $size - 1)
            printf(", ");

    }

    printf(") = [");

    for ($i = 0; $i < $size; $i++) {
        printf("%.6f", $outputs[$i]);
        if ($i < $size - 1)
            printf(", ");

    }

    printf("]\n");

    for ($i = 0; $i < $size; $i++) {
        printf("%.6f", $outputs[$i]);
        if ($i < $size - 1)
            printf(" + ");
    }

    $sum = array_sum($outputs);
    printf(" = %.1f\n", $sum);

    return 0;
}

exit(main($argc, $argv));

