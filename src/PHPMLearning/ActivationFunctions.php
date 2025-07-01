<?php
namespace PHPMLearning;

/**
 * 活性化関数に関するクラス
 */
class ActivationFunctions {

	/**
	 * ソフトマックス関数
     * @param float[] $x 入力値の配列
     * @return float[] ソフトマックス関数の出力値
	 */
	public static function softmax(array $x) : array {
		if (empty($x)) return [];
		$max = max($x);
		$exp_values = array_map(function($v) use ($max) { return exp($v - $max); }, $x);
		$sum_exp = array_sum($exp_values);
		return array_map(function($v) use ($sum_exp) { return $v / $sum_exp; }, $exp_values);
	}

	/**
	 * シグモイド関数
	 * @param float $x 入力値
	 * @return float シグモイド関数の出力値
	 */
	public static function sigmoid(float $x) : float {
		return 1 / (1 + exp(-$x));
	}

	/**
	 * シグモイド関数の微分
	 * @param float $y シグモイド関数の出力値
	 * @return float シグモイド関数の微分値
	 */
	public static function sigmoidDerivative(float $y) : float {
		return $y * (1 - $y);
	}

	/**
	 * ReLU関数 (Rectified Linear Unit)
	 * @param float $x 入力値
	 * @return float ReLU関数の出力値
	 */
	public static function relu(float $x) : float {
		return max(0, $x);
	}

	/**
	 * ReLU関数の微分
	 * @param float $x ReLU関数の入力値
	 * @return float ReLU関数の微分値
	 */
	public static function reluDerivative(float $x) : float {
		return $x > 0 ? 1 : 0;
	}

	/**
	 * tanh関数 (双曲線正接関数)
	 * @param float $x 入力値
	 * @return float tanh関数の出力値
	 */
	public static function tanh(float $x) : float {
		return (exp($x) - exp(-$x)) / (exp($x) + exp(-$x));
	}

	/**
	 * tanh関数の微分
	 * @param float $y tanh関数の出力値
	 * @return float tanh関数の微分値
	 */
	public static function tanhDerivative(float $y) : float {
		return 1 - $y * $y;
	}

	/**
	 * Leaky ReLU関数 (Leaky Rectified Linear Unit)
	 * @param float $x 入力値
	 * @param float $alpha 勾配の傾き（デフォルトは0.01）
	 * @return float Leaky ReLU関数の出力値
	 */
	public static function leakyrelU(float $x, float $alpha = 0.01) : float {
		return $x >= 0 ? $x : $alpha * $x;
	}

	/**
	 * Leaky ReLU関数の微分
	 * @param float $x Leaky ReLU関数の入力値
	 * @param float $alpha 勾配の傾き（デフォルトは0.01）
	 * @return float Leaky ReLU関数の微分値
	 */
	public static function leakyreluDerivative(float $x, float $alpha = 0.01) : float {
		return $x >= 0 ? 1 : $alpha;
	}

	/**
	 * ELU関数 (Exponential Linear Unit)
	 * @param float $x 入力値
	 * @param float $alpha 勾配の傾き（デフォルトは1.0）
	 * @return float ELU関数の出力値
	 */
	public static function elu(float $x, float $alpha = 1.0) : float {
		return $x >= 0 ? $x : $alpha * (exp($x) - 1);
	}

	/**
	 * ELU関数の微分
	 * @param float $x ELU関数の入力値
	 * @param float $alpha 勾配の傾き（デフォルトは1.0）
	 * @return float ELU関数の微分値
	 */
	public static function eluDerivative(float $x, float $alpha = 1.0) : float {
		return $x >= 0 ? 1 : $alpha * exp($x);
	}

	/**
	 * Swish関数
	 * @param float $x 入力値
	 * @return float Swish関数の出力値
	 */
	public static function swish(float $x) : float {
		return $x * self::sigmoid($x);
	}

	/**
	 * Swish関数の微分
	 * @param float $x Swish関数の入力値
	 * @return float Swish関数の微分値
	 */
	public static function swishDerivative(float $x) : float {
		$swish = self::swish($x);
		return $swish + self::sigmoid($x) * (1 - $swish);
	}

	/**
	 * GELU関数 (Gaussian Error Linear Unit)
	 * @param float $x 入力値
	 * @return float GELU関数の出力値
	 */
	public static function gelu(float $x) : float {
		$coeff = sqrt(2 / M_PI);
		$inner = $coeff * ($x + 0.044715 * pow($x, 3));
		return 0.5 * $x * (1 + tanh($inner));
	}

	/**
	 * GELU関数の微分
	 * @param float $x GELU関数の入力値
	 * @return float GELU関数の微分値
	 */
	public static function geluDerivative(float $x) : float {
		$coeff = sqrt(2 / M_PI);
		$inner = $coeff * ($x + 0.044715 * pow($x, 3));
		$tanh_inner  = tanh($inner);
		return 0.5 * (1 + $tanh_inner ) + 0.5 * $x * (1 - $tanh_inner  * $tanh_inner ) * $coeff * (1 + 3 * 0.044715 * pow($x, 2));
	}

	/**
	 * Softplus関数
	 * @param float $x 入力値
	 * @param float $max 最大値（デフォルトは20.0）
	 * @param float $min 最小値（デフォルトは-20.0）
	 * @return float Softplus関数の出力値
	 */
	public static function softplus(float $x, float $max = 20.0, float $min = -20.0) : float {
		if ($x > $max) return $x;
		if ($x < $min) return exp($x);
		
		return log(1 + exp($x));
	}

	/**
	 * Softplus関数の微分
	 * @param float $x Softplus関数の入力値
	 * @param float $max 最大値（デフォルトは20.0）
	 * @return float Softplus関数の微分値
	 */
	public static function softplusDerivative(float $x, float $max = 20.0) : float {
		if ($x > $max) return 1;
		return self::sigmoid($x);
	}

	/**
	 * 活性化関数の導関数のメタデータを取得
	 * @param string $funcname 活性化関数の名前
	 * @return array{output: bool} メタデータ（出力値かどうか）
	 */
	public static function getDerivativeMeta(string $funcname): array {
		return match ($funcname) {
			'sigmoid' => ['output' => true],
			'tanh' => ['output' => true],
			'relu' => ['output' => false],
			'leakyrelU' => ['output' => false],
			'elu' => ['output' => false],
			'swish' => ['output' => false],
			'gelu' => ['output' => false],
			'softplus' => ['output' => false],
			'softmax' => throw new \InvalidArgumentException("Use cross-entropy loss"),
			default => throw new \InvalidArgumentException("Unknown activation function: $funcname"),
		};
	}
	
	/**
	 * 活性化関数ごとの初期化スケールを取得
	 * @param string $funcname 活性化関数の名前
	 * @param int $n 入力の次元数
	 * @return float 初期化スケール
	 */
	public static function getInitScale(string $funcname, $n): float {
		if ($n == 0) return 1;
		return match ($funcname) {
			'sigmoid', 'tanh', 'softplus' => sqrt(1 / $n),
			'relu', 'leakyReLU', 'elu', 'swish', 'gelu' => sqrt(2 / $n),
			default => sqrt(1 / $n),
		};
	}

	public static function applyDerivative(string $funcname, float $z, float $a): float {
		$meta = self::getDerivativeMeta($funcname);
		$val = $meta['output'] ? $a : $z;
		$deriv_func = $funcname . 'Derivative';
		return self::$deriv_func($val);
	}
}
