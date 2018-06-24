package com.orange.plump.MultiPerceptron.brain;

import com.orange.plump.MultiPerceptron.math.Matrix;

public class NeuralNetworkFive {
	public int inputAmount, hidden1Amount, hidden2Amount, hidden3Amount, outputAmount;
	public Matrix weights_i1, bias_h1;
	public Matrix weights_12, bias_h2;
	public Matrix weights_23, bias_h3;
	public Matrix weights_3o, bias_o;
	public float learningRate;
	
	public NeuralNetworkFive(float learningRate, int inputs, int hidden1, int hidden2, int hidden3, int outputs) {
		this.inputAmount = inputs;
		this.hidden1Amount = hidden1;
		this.hidden2Amount = hidden2;
		this.hidden3Amount = hidden3;
		this.outputAmount = outputs;
		
		this.weights_i1 = new Matrix(this.hidden1Amount, this.inputAmount);
		this.weights_i1.randomize();
		this.bias_h1 = new Matrix(this.hidden1Amount, 1);
		this.bias_h1.randomize(1);
		
		this.weights_12 = new Matrix(this.hidden2Amount, this.hidden1Amount);
		this.weights_12.randomize();
		this.bias_h2 = new Matrix(this.hidden2Amount, 1);
		this.bias_h2.randomize(1);
		
		this.weights_23 = new Matrix(this.hidden3Amount, this.hidden2Amount);
		this.weights_23.randomize();
		this.bias_h3 = new Matrix(this.hidden3Amount, 1);
		this.bias_h3.randomize(1);
		
		this.weights_3o = new Matrix(this.outputAmount, this.hidden3Amount);
		this.weights_3o.randomize();
		this.bias_o = new Matrix(this.outputAmount, 1);
		this.bias_o.randomize(1);
		
		this.learningRate = learningRate;
	}
	
	public float[] feedFoward(float[] input) {
		if (input.length != inputAmount) return null;
		Matrix inputs = Matrix.fromArray(input);
		
		
		Matrix hidden1 = Matrix.matrixProduct(this.weights_i1, inputs);
		hidden1.add(this.bias_h1);
		hidden1.sigmoid();
		
		Matrix hidden2 = Matrix.matrixProduct(this.weights_12, hidden1);
		hidden2.add(this.bias_h2);
		hidden2.sigmoid();
		
		Matrix hidden3 = Matrix.matrixProduct(this.weights_23, hidden2);
		hidden3.add(this.bias_h3);
		hidden3.sigmoid();
		
		Matrix output = Matrix.matrixProduct(this.weights_3o, hidden3);
		output.add(this.bias_o);
		output.sigmoid();
		
		return Matrix.toArray(output);
	}
	
	public boolean train(float[] input, float[] targetArray) {
		//---
		
		Matrix inputs = Matrix.fromArray(input);
		
		
		Matrix hidden1 = Matrix.matrixProduct(this.weights_i1, inputs);
		hidden1.add(this.bias_h1);
		hidden1.sigmoid();
		
		Matrix hidden2 = Matrix.matrixProduct(this.weights_12, hidden1);
		hidden2.add(this.bias_h2);
		hidden2.sigmoid();
		
		Matrix hidden3 = Matrix.matrixProduct(this.weights_23, hidden2);
		hidden3.add(this.bias_h3);
		hidden3.sigmoid();
		
		Matrix output = Matrix.matrixProduct(this.weights_3o, hidden3);
		output.add(this.bias_o);
		output.sigmoid();
		//---
		
		Matrix targets = Matrix.fromArray(targetArray);
		

		//Finds Output Errors
		Matrix output_error = Matrix.subtract(targets, output);
		//Finds The Output Gradient
		Matrix output_gradient = Matrix.dSigmoid(output);
		
		output_gradient.multiply(output_error);
		output_gradient.multiply(learningRate);
		
		//Transposes Hidden Layer
		Matrix hidden3T = Matrix.transpose(hidden3);
		//Creates Weights Deltas With The Output Gradient And The Transposed Hidden Layer
		Matrix weights_3o_del = Matrix.matrixProduct(output_gradient, hidden3T);
		//Adds The Delta To The Weights
		this.weights_3o.add(weights_3o_del);
		this.bias_o.add(output_gradient);
		//Transposes Hidden->Output Weights
		Matrix weights_3o_t = Matrix.transpose(this.weights_3o);
		//Finds The Errors From The Hidden Layer With THe Transposed Layer
		Matrix hidden3_error = Matrix.matrixProduct(weights_3o_t, output_error);
		
		//Finds The Hidden Layers Gradient
		Matrix hidden3_gradient = Matrix.dSigmoid(hidden3);
		
		hidden3_gradient.multiply(hidden3_error);
		hidden3_gradient.multiply(learningRate);
		
		//Transposes Hidden Layer
		Matrix hidden2T = Matrix.transpose(hidden2);
		//Creates Weights Deltas With The Output Gradient And The Transposed Hidden Layer
		Matrix weights_23_del = Matrix.matrixProduct(hidden3_gradient, hidden2T);
		//Adds The Delta To The Weights
		this.weights_23.add(weights_23_del);
		this.bias_h3.add(hidden3_gradient);
		//Transposes Hidden->Output Weights
		Matrix weights_23_t = Matrix.transpose(this.weights_23);
		//Finds The Errors From The Hidden Layer With THe Transposed Layer
		Matrix hidden2_error = Matrix.matrixProduct(weights_23_t, hidden3_error);
		
		//Finds The Hidden Layers Gradient
		Matrix hidden2_gradient = Matrix.dSigmoid(hidden2);
		
		hidden2_gradient.multiply(hidden2_error);
		hidden2_gradient.multiply(learningRate);
		
		//Transposes Hidden Layer
		Matrix hidden1T = Matrix.transpose(hidden1);
		//Creates Weights Deltas With The Output Gradient And The Transposed Hidden Layer
		Matrix weights_12_del = Matrix.matrixProduct(hidden2_gradient, hidden1T);
		//Adds The Delta To The Weights
		this.weights_12.add(weights_12_del);
		this.bias_h2.add(hidden2_gradient);
		//Transposes Hidden->Output Weights
		Matrix weights_12_t = Matrix.transpose(this.weights_12);
		//Finds The Errors From The Hidden Layer With THe Transposed Layer
		Matrix hidden1_error = Matrix.matrixProduct(weights_12_t, hidden2_error);
		
		//Finds The Hidden Layers Gradient
		Matrix hidden1_gradient = Matrix.dSigmoid(hidden1);
		
		hidden1_gradient.multiply(hidden1_error);
		hidden1_gradient.multiply(learningRate);
				
		//Transposes The Input Layer
		Matrix inputT = Matrix.transpose(inputs);
		//Creates Weights Deltas With The Hidden Gradient And THe Transposed Input Layer
		Matrix weights_i1_del = Matrix.matrixProduct(hidden1_gradient, inputT);
		//Adds The Delta To The Weights
		this.weights_i1.add(weights_i1_del);
		this.bias_h1.add(hidden1_gradient);
		
		float[] outputs = Matrix.toArray(output);
		int total = 0;
		for (int i = 0; i < targetArray.length; i++) {
			if (targetArray[i] == 1f) {
				if (outputs[i] > 0.5f) {
					total++;
				}
			} else {
				if (outputs[i] < 0.5f) {
					total++;
				}
			}
		}
		if (total == targetArray.length) {
			return true;
		} else return false;
	}
}
