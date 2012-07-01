(ns Feed-Forward-Neuralnet.learning)

(defn continue-learning-function
  [err]
  (let [threshold 0.001]
    (if (> err threshold)
      t
      nil)))

(defn perceptron-learning [perceptron err examples continue-learning-fn]
  "Perceptron learning implemented from AIMA vol. 2
    function PERCEPTRON-LEARNING(examples, network) -> a perceptron hypothesis
        examples: a set of examples, each with input vector x (x1, ..., xn) and output y
        network, a perceptron with weights Wj, j = 0..n, and activation function g

    while should_learn:
        for e in examples
            in = SUM(j=0, n, Wj*xj[e])
            Err = y[e] - g(in)
            Wj = Wj + alpha * Err * g'(in) x xj[e]
    return nnhypothesis(network)

  notes:
     alpha is the learning rate of this algorithm
     you can see that when the Err is positive then the network output is too small,
     the weights are increasd for 
  "
  (loop [perceptron error]
    (if (not (continue-learning error))
      perceptron
      (let [perceptron_error (update-perceptron perceptron error)
            perceptron (first perceptron_error)
            error (second perceptron_error)]
        (recur perceptron error)))))

(defn set-weights
  [perceptron new-weights]
  (assoc perceptron :weights new-weights))

(defn update-perceptron
  [perceptron error]
  (for [example (examples perceptron)]
      (let [in (sum-over-weights network example)
            err (calc-err (ex-output example) (activation-fn perceptron))
            new-weights (update-weights
                         (weights perceptron) alpha err (activation-prime perceptron) example)]
        [(set-weights perceptron new-weights) err])))