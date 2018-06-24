package com.orange.plump.MultiPerceptron.math;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Matrix {
	
	public final int rowAmount;
	public final int colAmount;
	public List<List<Float>> matrix;

	// Takes In 2 Integers
	// rows = Matrixes' Row Amount
	// cols = Matrixes' Col Amount
	//
	public Matrix(int rows, int cols) {
		this.rowAmount = rows;
		this.colAmount = cols;
		this.matrix = new ArrayList<List<Float>>();
		
		createLocalMatrix();
	}
	
	//Creates The Empty Matrix
	private void createLocalMatrix() {
		for (int i = 0; i < rowAmount; i++) {
			List<Float> row = new ArrayList<Float>();
			for (int j = 0; j < colAmount; j++) 
				row.add(0f);
			matrix.add(row);
		}
	}
	
	//Prints The Matrix In A Graphed Fashion
	public void print() {
		for (List<Float> row : this.matrix) {
			String s = "";
			for (float i : row) {
				s += i + " ";
			}
			System.out.println(s + "\n");
		}
		System.out.println("\n");
	}
	
	public void randomize(int n) {
		for (int i = 0; i < rowAmount; i++)
			for (int j = 0; j < colAmount; j++) {
				Random ran = new Random();
				matrix.get(i).set(j, (ran.nextInt(n * 2 + 1) - n) * 1f);
			}
	}
	
	public void randomize() {
		for (int i = 0; i < rowAmount; i++)
			for (int j = 0; j < colAmount; j++) {
				Random ran = new Random();
				matrix.get(i).set(j, (ran.nextInt(3) - 1) * 1f);
			}
	}
	
	public void sigmoid() {
		for (int i = 0; i < rowAmount; i++)
			for (int j = 0; j < colAmount; j++)
				matrix.get(i).set(j, (float) (1 / (1 + Math.exp(-1 * matrix.get(i).get(j)))));
	}
	
	public static Matrix dSigmoid(Matrix mat) {
		Matrix result = new Matrix(mat.rowAmount, mat.colAmount);
		for (int i = 0; i < mat.rowAmount; i++)
			for (int j = 0; j < mat.colAmount; j++)
				result.matrix.get(i).set(j, mat.matrix.get(i).get(j) * (1 - mat.matrix.get(i).get(j)));
		return result;
	}
	
	public static Matrix sigmoid(Matrix mat) {
		Matrix result = new Matrix(mat.rowAmount, mat.colAmount);
		for (int i = 0; i < mat.rowAmount; i++)
			for (int j = 0; j < mat.colAmount; j++)
				result.matrix.get(i).set(j, (float) (1 / (1 + Math.exp(-1 * mat.matrix.get(i).get(j).doubleValue()))));
		return result;
	}
	
	// Scalar Addition
	// |2, 3|       |3, 4|
	// |6, 5| + 1 = |7, 6|
	//
	public void add(float amount) {
		for (int i = 0; i < rowAmount; i++)
			for (int j = 0; j < colAmount; j++)
				matrix.get(i).set(j, matrix.get(i).get(j) + amount);
	}
	
	public void subtract(float amount) {
		for (int i = 0; i < rowAmount; i++)
			for (int j = 0; j < colAmount; j++)
				matrix.get(i).set(j, matrix.get(i).get(j) - amount);
	}
	// Scalar Multiplication
	// |2, 3|       |4, 6|
	// |3, 1| x 2 = |6, 2|
	//
	public void multiply(float amount) {
		for (int i = 0; i < rowAmount; i++)
			for (int j = 0; j < colAmount; j++)
				matrix.get(i).set(j, matrix.get(i).get(j) * amount);
	}
	
	public void divide(float amount) {
		for (int i = 0; i < rowAmount; i++)
			for (int j = 0; j < colAmount; j++)
				matrix.get(i).set(j, matrix.get(i).get(j) / amount);
	}
	
	// Element-Wise Addition
	// |2, 3|   |1, 1|   |3, 4|
	// |6, 5| + |1, 1| = |7, 6|
	//
	// Returns False If Matrixes' Size's Aren't Compatible
	public boolean add(Matrix mat) {
		if (mat.rowAmount != rowAmount || mat.colAmount != colAmount) return false;
		float amount = 0;
		for (int i = 0; i < rowAmount; i++)
			for (int j = 0; j < colAmount; j++) {
				amount = mat.matrix.get(i).get(j);
				matrix.get(i).set(j, matrix.get(i).get(j) + amount);
			}
		return true;
	}

	public boolean subtract(Matrix mat) {
		if (mat.rowAmount != rowAmount || mat.colAmount != colAmount) return false;
		float amount = 0;
		for (int i = 0; i < rowAmount; i++)
			for (int j = 0; j < colAmount; j++) {
				amount = mat.matrix.get(i).get(j);
				matrix.get(i).set(j, matrix.get(i).get(j) - amount);
			}
		return true;
	}
	
	// Element-Wise Multiplication
	// |2, 3|   |2, 2|   |4, 6|
	// |3, 1| x |2, 2| = |6, 2|
	//
	// Returns False If Matrixes' Size's Aren't Compatible
	public boolean multiply(Matrix mat) {
		if (mat.rowAmount != rowAmount || mat.colAmount != colAmount) return false;
		float amount = 0;
		for (int i = 0; i < rowAmount; i++)
			for (int j = 0; j < colAmount; j++) {
				amount = mat.matrix.get(i).get(j);
				matrix.get(i).set(j, matrix.get(i).get(j) * amount);
			}
		return true;
	}
	
	public boolean divide(Matrix mat) {
		if (mat.rowAmount != rowAmount || mat.colAmount != colAmount) return false;
		float amount = 0;
		for (int i = 0; i < rowAmount; i++)
			for (int j = 0; j < colAmount; j++) {
				amount = mat.matrix.get(i).get(j);
				matrix.get(i).set(j, matrix.get(i).get(j) / amount);
			}
		return true;
	}
	
	// Element-Wise Multiplication
	// |2, 3|   |2, 2|   |4, 6|
	// |3, 1| x |2, 2| = |6, 2|
	//
	// Returns New Matrix
	public static Matrix multiply(Matrix mat1, Matrix mat2) {
		
		if (mat2.rowAmount != mat1.rowAmount || mat2.colAmount != mat1.colAmount) return null;
		Matrix result = new Matrix(mat1.rowAmount, mat1.colAmount);
		float amount = 0;
		for (int i = 0; i < mat1.rowAmount; i++)
			for (int j = 0; j < mat1.colAmount; j++) {
				amount = mat2.matrix.get(i).get(j);
				result.matrix.get(i).set(j, mat1.matrix.get(i).get(j) * amount);
			}
		return result;
	}
	
	public static Matrix divide(Matrix mat1, Matrix mat2) {
		
		if (mat2.rowAmount != mat1.rowAmount || mat2.colAmount != mat1.colAmount) return null;
		Matrix result = new Matrix(mat1.rowAmount, mat1.colAmount);
		float amount = 0;
		for (int i = 0; i < mat1.rowAmount; i++)
			for (int j = 0; j < mat1.colAmount; j++) {
				amount = mat2.matrix.get(i).get(j);
				result.matrix.get(i).set(j, mat1.matrix.get(i).get(j) / amount);
			}
		return result;
	}
	
	// Element-Wise Multiplication
	// |2, 3|   |2, 2|   |4, 6|
	// |3, 1| x |2, 2| = |6, 2|
	//
	// Returns New Matrix
	public static Matrix add(Matrix mat1, Matrix mat2) {
		
		if (mat2.rowAmount != mat1.rowAmount || mat2.colAmount != mat1.colAmount) return null;
		Matrix result = new Matrix(mat1.rowAmount, mat1.colAmount);
		float amount = 0;
		for (int i = 0; i < mat1.rowAmount; i++)
			for (int j = 0; j < mat1.colAmount; j++) {
				amount = mat2.matrix.get(i).get(j);
				result.matrix.get(i).set(j, mat1.matrix.get(i).get(j) + amount);
			}
		return result;
	}
	
	public static Matrix subtract(Matrix mat1, Matrix mat2) {
		
		if (mat2.rowAmount != mat1.rowAmount || mat2.colAmount != mat1.colAmount) return null;
		Matrix result = new Matrix(mat1.rowAmount, mat1.colAmount);
		float amount = 0;
		for (int i = 0; i < mat1.rowAmount; i++)
			for (int j = 0; j < mat1.colAmount; j++) {
				amount = mat2.matrix.get(i).get(j);
				result.matrix.get(i).set(j, mat1.matrix.get(i).get(j) - amount);
			}
		return result;
	}
	
	public static Matrix fromArray(float[] args) {
		Matrix result = new Matrix(args.length, 1);
		
		for (int i = 0; i < result.rowAmount; i++)
			result.matrix.get(i).set(0, args[i]);
		return result;
	}
	
	public static float[] toArray(Matrix mat) {
		float[] result = new float[mat.rowAmount];
		
		for (int i = 0; i < mat.rowAmount; i++)
			for (int j = 0; j < mat.colAmount; j++)
			result[i] = mat.matrix.get(i).get(0);
		return result;
	}
	
	// Matrix Product
	// |2, 3, 4|   |2, 2|   |18, 18|  
	// |3, 1, 1| . |2, 2| = |10, 10|
	//             |2, 2|
	// Returns The Product
	public static Matrix matrixProduct(Matrix A, Matrix B) {
		if (B.rowAmount != A.colAmount) return null;
		
		Matrix result = new Matrix(A.rowAmount, B.colAmount);
		
		for (int i = 0; i < result.rowAmount; i++)
			for (int j = 0; j < result.colAmount; j++) {
				float sum = 0;
				for (int k = 0; k < A.colAmount; k++) {
					sum += A.matrix.get(i).get(k) * B.matrix.get(k).get(j);
				}
				result.matrix.get(i).set(j, sum);
			}
		
		
		return result;
	}
	
	// Matrix Transposing
	// |2, 3, 4|    |2, 3|
	// |3, 1, 1| -> |3, 1|
	//              |4, 1|
	// Returns The Transposed Matrix
	public static Matrix transpose(Matrix A) {
		Matrix result = new Matrix(A.colAmount, A.rowAmount);
		for (int i = 0; i < A.rowAmount; i++)
			for (int j = 0; j < A.colAmount; j++)
				result.matrix.get(j).set(i, A.matrix.get(i).get(j));
		return result;
	}
	
	public static boolean areSameSize(Matrix A, Matrix B) {
		if (A.rowAmount == B.rowAmount && A.colAmount == B.colAmount) return true;
		return false;
	}
}
