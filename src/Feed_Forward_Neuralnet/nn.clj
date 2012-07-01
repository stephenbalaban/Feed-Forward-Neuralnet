(ns Feed-Forward-Neuralnet.nn)

(defrecord NeuralNet
    [weights activation-fn activation-fn-prime])