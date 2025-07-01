<?php
namespace PHPMLearning;

use PHPMLearning\ActivationFunctions;
use Random\Engine\Mt19937;
use Random\Randomizer;

class NeuralNetwork {
	private int $seed = -1;              // 乱数のシード値（-1ならランダム）
	private int $num_layers;	         // 層の総数 (例: 4)

	/** @var int[] $layer_neuron_counts */
	private array $layer_neuron_counts;  // 各層のニューロン数の配列 (例: [7, 10, 8, 3])

	/** @var float[][][] $weights */
	private array $weights;			     // 重み (層ごとの配列)

	/** @var float[][] $biases */
	private array $biases;			     // バイアス (層ごとの配列)

	/** @var float[][] $activations */
	private array $activations;		     // 各層の出力値（フォワード時に計算）

	/** @var array<int, string> */
	private array $activation_functions; // 各層の活性化関数 (例: 'sigmoid', 'relu', 'tanh', ...)

	/** @var float[][] $z_values */
	private array $z_values;

	private string $loss_function = 'mse'; // mse (平均二乗誤差)、cross_entropy (交差エントロピー) 

	/**
	 * コンストラクタ
	 * @param int|int[] $hidden_neurons 中間層のニューロン数（単一の整数または配列で指定）
	 * @param int $input_neurons 入力層のニューロン数
	 * @param int $output_neurons 出力層のニューロン数
	 * @param string $loss_function 損失関数名 ('mse' または 'cross_entropy')
	 * @param int $seed 乱数のシード値（-1ならランダム）
	 */
	public function __construct(int | array $hidden_neurons = [8, 6], int $input_neurons = 10, int $output_neurons = 4, string $loss_function = 'mse', int $seed = -1) {
		if (!is_array($hidden_neurons)) // 単一の中間層ニューロン数が指定された場合は配列に変換
			$hidden_neurons = [$hidden_neurons];
		
		$this->seed = $seed;
		$this->loss_function = $loss_function;

		$this->layer_neuron_counts = array_merge([$input_neurons], $hidden_neurons, [$output_neurons]); // 入力層、中間層、出力層のニューロン数を配列で定義
		$this->num_layers = count($this->layer_neuron_counts);

		$this->weights = [];
		$this->biases = [];
		$this->activations = [];
		
		// 全層の活性化関数をシグモイド関数をデフォルトに
		$this->activation_functions = array_fill(0, $this->num_layers - 1, 'sigmoid');
	}

	/**
	 * 活性化関数を設定
	 * @param int $index 層のインデックス
	 * @param string $funcname 活性化関数の名前
	 */
	public function setActivationFunction(int $index, string $funcname): void {
		if ($index < 0 || $index >= $this->num_layers - 1)
			throw new \OutOfRangeException("Layer index out of range.");
		
		$this->activation_functions[$index] = $funcname;
	}

	/**
	 * 入力層のニューロン数を設定
	 * @param int $n 入力層のニューロン数
	 */
	public function setInputNeurons(int $n): void {
		$this->layer_neuron_counts[0] = $n;
	}

	/**
	 * 中間層のニューロン数を設定
	 * @param int $index 中間層のインデックス
	 * @param int $n 中間層のニューロン数
	 */
	public function setHiddenNeurons(int $index, int $n): void {
		if ($index < 0 || $index >= $this->num_layers - 2)
			throw new \OutOfRangeException("Hidden layer index out of range.");
		
		$this->layer_neuron_counts[$index + 1] = $n;
	}

	/**
	 * 出力層のニューロン数を設定
	 * @param int $n 出力層のニューロン数
	 */
	public function setOutputNeurons(int $n): void {
		$this->layer_neuron_counts[$this->num_layers - 1] = $n;
	}

	/**
	 * 乱数のシード値を設定
	 * @param int $seed シード値
	 */
	public function setSeed(int $seed): void {
		$this->seed = $seed;
	}

	/**
	 * 重みとバイアスをランダムに初期化する
	 */
	public function initWeights(): void {
		if ($this->seed === -1)
			$this->seed = mt_rand();

		
		$engine = new Mt19937($this->seed);
		$rand = new Randomizer($engine);
		
		/** @phpstan-ignore-next-line */
		$use_mt19937 = method_exists($rand, 'getFloat');
		
		if (!$use_mt19937)
			mt_srand($this->seed);

		// 層の接続ごとにループ (入力層から最後から2番目の層まで)
		for ($i = 0; $i < $this->num_layers - 1; $i++) {
			$from_neurons = $this->layer_neuron_counts[$i];
			$to_neurons = $this->layer_neuron_counts[$i+1];

			$scale = ActivationFunctions::getInitScale($this->activation_functions[$i], $from_neurons);

			// 重み行列を初期化
			$this->weights[$i] = [];
			for ($j = 0; $j < $from_neurons; $j++) {
				for ($k = 0; $k < $to_neurons; $k++) {
					if ($use_mt19937)
						$this->weights[$i][$j][$k] = $rand->getFloat(-1, 1) * $scale;
					else 
						$this->weights[$i][$j][$k] = ((mt_rand() / mt_getrandmax()) * 2 - 1) * $scale;
				}
			}

			// バイアスを初期化
			$this->biases[$i] = [];
			for ($k = 0; $k < $to_neurons; $k++) {
				$this->biases[$i][$k] = 0.0;
			}
		}
	}

	/**
	 * モデルのデータを読み込む
	 * @param array<string, mixed> $model モデルデータ
	 * @throws \InvalidArgumentException モデルデータが不正な場合
	 */
	public function loadModel(array $model): void {
		if (!isset($model['weights']) || !isset($model['biases'])) {
			throw new \InvalidArgumentException("Invalid model data: weights or biases are missing.");
		}

		if (isset($model['seed']))
			$this->seed = $model['seed'];

		if (isset($model['num_layers'])) 
			$this->num_layers = $model['num_layers'];

		if (isset($model['layer_neuron_counts'])) {
			$this->layer_neuron_counts = $model['layer_neuron_counts'];
			if (!isset($model['num_layers']))
				$this->num_layers = count($this->layer_neuron_counts);
		}

		$this->weights = $model['weights'];
		$this->biases = $model['biases'];

		if (isset($model['activation_functions']))
			$this->activation_functions = $model['activation_functions'];
		
		if (isset($model['loss_function']))
			$this->loss_function = $model['loss_function'];
		
	}

	/**
	 * モデルのデータを取得
	 * @return array<string, mixed> モデルのデータ
	 */
	public function getModel() : array {
		return [
			'seed' => $this->seed,
			'layer_neuron_counts' => $this->layer_neuron_counts,
			'num_layers' => $this->num_layers,
			'weights' => $this->weights,
			'biases' => $this->biases,
			'activation_functions' => $this->activation_functions,
			'loss_function' => $this->loss_function
		];
	}

	/**
	 * フォワードプロパゲーション
	 * @param float[] $input 入力データ
	 * @return float[][] 全ての層の活性化値（ニューロンの出力値）
	 */
	public function forward(array $input) : array {
		$this->activations = [$input];
		$current_activations = $input; // 現在の活性化値を入れるところ、最初は入力データをセット

		$this->z_values = [];

		// 層の接続ごとにループ
		for ($i = 0; $i < $this->num_layers - 1; $i++) {
			$to_neurons = $this->layer_neuron_counts[$i+1];
			$from_neurons = $this->layer_neuron_counts[$i];

			$layer_z_values = [];
			for ($k = 0; $k < $to_neurons; $k++) {
				$sum = $this->biases[$i][$k];
				for ($j = 0; $j < $from_neurons; $j++) {
					$sum += $current_activations[$j] * $this->weights[$i][$j][$k];
				}
				$layer_z_values[$k] = $sum;
			}

			$next_activations = []; // 次の層の活性化値を入れる配列

			$funcname = $this->activation_functions[$i];
			if ($funcname === 'softmax') {
				// softmaxの場合、計算したロジットの配列全体を渡す
				$next_activations = ActivationFunctions::softmax($layer_z_values);
			} else {
				// 他の活性化関数の場合、各ロジットに個別に適用
				foreach ($layer_z_values as $z) {
					$next_activations[] = ActivationFunctions::$funcname($z);
				}
			}
			
			$this->z_values[] = $layer_z_values;
			$this->activations[] = $next_activations;
			$current_activations = $next_activations;
		}

		return $this->activations;
	}

	/**
	 * バックプロパゲーションで重み更新（学習）
	 * @param float[] $input 入力特徴量の配列
	 * @param float[] $target 目標出力の配列
	 * @param float $learning_rate 学習率
	 * @return float 平均二乗誤差
	 */
	public function trainNonBatch(array $input, array $target, float $learning_rate) : float {
		return $this->train([['features' => $input, 'label' => $target]], $learning_rate);
	}

	/**
	 * ミニバッチを使ってバックプロパゲーションで重み更新（学習）
	 * @param array<int, array{features: float[], label: float[]}> $batch データ(featuresとlabelの配列)の配列
	 * @param float $learning_rate 学習率
	 * @return float バッチ全体の平均二乗誤差
	 * @throws \InvalidArgumentException バッチが空の場合
	 * @throws \RuntimeException フォワードプロパゲーションで出力が得られなかった場合
	 */
	public function train(array $batch, float $learning_rate) : float {
		if (empty($batch))
			throw new \InvalidArgumentException("Batch cannot be empty.");

		$batch_size = count($batch);

		// バッチ全体のデルタの合計を溜め込むための変数を初期化
		$total_weight_deltas = $this->weights;
		// すべて0で埋める
		array_walk_recursive($total_weight_deltas, fn(&$v) => $v=0);
		$total_bias_deltas = $this->biases;
		array_walk_recursive($total_bias_deltas, fn(&$v) => $v=0);

		$total_mse = 0;

		// バッチ内の各データでデルタを計算し、合計していく
		foreach ($batch as $data) {
			$input = $data['features'];
			$target = $data['label'];

			// デルタ(誤差)計算
			$all_activations = $this->forward($input);
			$output = end($all_activations);
			if (empty($output))
				throw new \RuntimeException("Forward propagation did not produce output.");

			$deltas = array_fill(0, $this->num_layers - 1, []);
			
			// 出力層のデルタ
			$last_layer_idx = $this->num_layers - 1;
			$output_deltas = [];
			
			// 出力層が 'softmax' かつ 損失関数が 'cross_entropy' の場合
			if ($this->activation_functions[$last_layer_idx - 1] === 'softmax' && $this->loss_function === 'cross_entropy') {
				for ($k = 0; $k < $this->layer_neuron_counts[$last_layer_idx]; $k++) {
					// デルタは (予測確率 - 正解ラベル) となる
					$output_deltas[$k] = $output[$k] - $target[$k];
				}
			} else {
				// それ以外の場合（MSEなど）は、従来の計算
				for ($k = 0; $k < $this->layer_neuron_counts[$last_layer_idx]; $k++) {
					$error = $target[$k] - $output[$k];
					$z = $this->z_values[$last_layer_idx - 1][$k];
					$a = $output[$k];

					$output_deltas[$k] = $error * ActivationFunctions::applyDerivative(
						$this->activation_functions[$last_layer_idx - 1],
						$z,
						$a
					);
				}
			}
			$deltas[$last_layer_idx - 1] = $output_deltas;
			
			// 中間層のデルタ
			for ($i = $this->num_layers - 2; $i > 0; $i--) {
				$prev_layer_deltas = [];
				$next_layer_deltas = $deltas[$i];
				$weights_to_next = $this->weights[$i];
				for ($j = 0; $j < $this->layer_neuron_counts[$i]; $j++) {
					$error = 0.0;
					for ($k = 0; $k < $this->layer_neuron_counts[$i + 1]; $k++) {
						$error += $next_layer_deltas[$k] * $weights_to_next[$j][$k];
					}
					
					$z = $this->z_values[$i - 1][$j];
					$a = $all_activations[$i][$j];
					$prev_layer_deltas[$j] = $error * ActivationFunctions::applyDerivative(
						$this->activation_functions[$i - 1],
						$z,
						$a
					);

				}
				$deltas[$i - 1] = $prev_layer_deltas;
			}

			// 計算したデルタを合計デルタに加算
			for ($i = 0; $i < $this->num_layers - 1; $i++) {
				for ($j = 0; $j < $this->layer_neuron_counts[$i]; $j++) {
					for ($k = 0; $k < $this->layer_neuron_counts[$i+1]; $k++) {
						$total_weight_deltas[$i][$j][$k] += $deltas[$i][$k] * $all_activations[$i][$j];
					}
				}
				for ($k = 0; $k < $this->layer_neuron_counts[$i+1]; $k++) {
					$total_bias_deltas[$i][$k] += $deltas[$i][$k];
				}
			}

			// lossも計算しておく
			$total_mse += $this->computeLoss($target, $output);
		}

		// バッチ全体の平均デルタを使って、重みとバイアスを更新
		for ($i = 0; $i < $this->num_layers - 1; $i++) {
			for ($j = 0; $j < $this->layer_neuron_counts[$i]; $j++) {
				for ($k = 0; $k < $this->layer_neuron_counts[$i+1]; $k++) {
					$this->weights[$i][$j][$k] -= $learning_rate * ($total_weight_deltas[$i][$j][$k] / $batch_size);
				}
			}
			for ($k = 0; $k < $this->layer_neuron_counts[$i+1]; $k++) {
				$this->biases[$i][$k] -= $learning_rate * ($total_bias_deltas[$i][$k] / $batch_size);
			}
		}

		return $total_mse / $batch_size; // トータル平均二乗誤差 / バッチ数 = 損失度合
	}

	/**
	 * 損失関数を計算
	 * @param float[] $target 目標出力の配列
	 * @param float[] $output モデルの出力値の配列
	 * @return float 損失値
	 */
	public function computeLoss(array $target, array $output) : float {
		$size = count($output);

		// 平均二乗誤差 (MSE)
		if ($this->loss_function === 'mse') {
			$sum = 0.0;
			for ($i = 0; $i < $size; $i++) {
				$sum += pow($target[$i] - $output[$i], 2);
			}
			return $sum / $size;
		}

		// 交差エントロピー損失
		if ($this->loss_function === 'cross_entropy') {
			$sum = 0.0;
			for ($i = 0; $i < $size; $i++) {
				$sum -= $target[$i] * log(max($output[$i], 1e-10));
			}
			return $sum / $size;
		}

		throw new \RuntimeException("Unsupported loss function: " . $this->loss_function);
	}

	/** 
	 * 予測ラベルと確率の配列を取得する (推論)
	 * @param float[] $input 特徴量の配列
	 * @return array{label: int, probs: float[]} 予測ラベルと確率の配列
	 * @throws \RuntimeException 確率を生成しなかった場合
	 */ 
	public function predict(array $input) : array {
		$output = $this->getOutput($input);

		$probs = ($this->activation_functions[$this->num_layers - 2] === 'softmax')
				? $output :	ActivationFunctions::softmax($output);
		
		if (empty($probs))
			throw new \RuntimeException("Prediction did not produce probabilities.");

		// 最大の確率を持つインデックスを取得
		$max_index = array_keys($probs, max($probs))[0];
		
		// ラベルは出力層のニューロン数に応じて決定
		return [
			'label' => $max_index, // 予測ラベル
			'probs' => $probs // 確率の配列
		];
	}

	/**
	 * 入力データに対する出力を取得
	 * @param float[] $input 特徴量の配列
	 * @return float[] 最後の層の出力値
	 * @throws \RuntimeException フォワードプロパゲーションで出力が得られなかった場合
	 */
	public function getOutput(array $input) : array {
		$all_activations = $this->forward($input);
		if (empty($all_activations))
			throw new \RuntimeException("Forward propagation did not produce activations.");
		return end($all_activations); // 最後の層の出力が最終結果
	}

	/**
	 * 確率の配列を返す
	 * @param float[] $input 特徴量の配列
	 * @return float[] 確率の配列
	 */
	public function getProbs(array $input) : array {
		$output = $this->getOutput($input);

		$probs = ($this->activation_functions[$this->num_layers - 2] === 'softmax')
				? $output :	ActivationFunctions::softmax($output);
		return $probs;
	}
}
