package com.orange.plump.MultiPerceptron.brain;

import com.orange.plump.MultiPerceptron.math.Matrix;

public class NeuralNetworkThree {
	
	public int inputAmount, hiddenAmount, outputAmount;
	public Matrix weights_ih, bias_h;
	public Matrix weights_ho, bias_o;
	public float learningRate;
	
	public NeuralNetworkThree(float learningRate, int inputs, int hiddens, int outputs) {
		this.inputAmount = inputs;
		this.hiddenAmount = hiddens;
		this.outputAmount = outputs;
		
		this.weights_ih = new Matrix(this.hiddenAmount, this.inputAmount);
		this.weights_ho = new Matrix(this.outputAmount, this.hiddenAmount);
		this.weights_ih.randomize();
		this.weights_ho.randomize();
		
		this.bias_h = new Matrix(this.hiddenAmount, 1);
		this.bias_o = new Matrix(this.outputAmount, 1);
		this.bias_h.randomize(1);
		this.bias_o.randomize(1);
		this.learningRate = learningRate;
	}
	
	public float[] feedFoward(float[] input) {
		if (input.length != inputAmount) return null;
		Matrix inputs = Matrix.fromArray(input);
		
		
		Matrix hidden = Matrix.matrixProduct(this.weights_ih, inputs);
		hidden.add(this.bias_h);
		hidden.sigmoid();
		
		Matrix output = Matrix.matrixProduct(this.weights_ho, hidden);
		output.add(this.bias_o);
		output.sigmoid();
		
		return Matrix.toArray(output);
	}
	
	public void train(float[] input, float[] targetArray) {
		//---
		if (input.length != inputAmount) return;
		Matrix inputs = Matrix.fromArray(input);
		
		
		Matrix hidden = Matrix.matrixProduct(this.weights_ih, inputs);
		hidden.add(this.bias_h);
		hidden.sigmoid();
		
		Matrix output = Matrix.matrixProduct(this.weights_ho, hidden);
		output.add(this.bias_o);
		output.sigmoid();
		//---
		
		Matrix targets = Matrix.fromArray(targetArray);
		
		if(!(Matrix.areSameSize(output, targets))) return;
		//Finds Output Errors
		Matrix output_error = Matrix.subtract(targets, output);
		//Finds The Output Gradient
		Matrix gradient = Matrix.dSigmoid(output);
		
		gradient.multiply(output_error);
		gradient.multiply(learningRate);
		
		//Transposes Hidden Layer
		Matrix hiddenT = Matrix.transpose(hidden);
		//Creates Weights Deltas With The Output Gradient And The Transposed Hidden Layer
		Matrix weights_ho_del = Matrix.matrixProduct(gradient, hiddenT);
		//Adds The Delta To The Weights
		this.weights_ho.add(weights_ho_del);
		this.bias_o.add(gradient);
		//Transposes Hidden->Output Weights
		Matrix weights_ho_t = Matrix.transpose(this.weights_ho);
		//Finds The Errors From The Hidden Layer With THe Transposed Layer
		Matrix hidden_error = Matrix.matrixProduct(weights_ho_t, output_error);
		
		//Finds The Hidden Layers Gradient
		Matrix hidden_gradient = Matrix.dSigmoid(hidden);
		
		hidden_gradient.multiply(hidden_error);
		hidden_gradient.multiply(learningRate);
		
		//Transposes The Input Layer
		Matrix inputT = Matrix.transpose(inputs);
		//Creates Weights Deltas With The Hidden Gradient And THe Transposed Input Layer
		Matrix weights_ih_del = Matrix.matrixProduct(hidden_gradient, inputT);
		//Adds The Delta To The Weights
		this.weights_ih.add(weights_ih_del);
		this.bias_h.add(hidden_gradient);
		
	}
}
