from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(20,),
                    max_iter=1000)
mlp.fit(X_train,y_train)
print("MLP Accuracy =",mlp.score(X_test,y_test))
