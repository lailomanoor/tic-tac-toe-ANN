#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <string>
using namespace std;

struct Connection
{
    double weight;
    double deltaWeight;
};

class neuron; // forward declaration of class so we can declare Layer

typedef vector<neuron> Layer; // vector of neurons

// ****************** class neuron ***********************//
class neuron
{
private:
    double m_outputVal;
    unsigned m_myIndex;
    double m_gradient;

    static double eta; // {0.0...0.1} overall net training rate
                       /* 0.0 slow learner
                          0.2 medium learner
                          1.0 reckless learner */

    static double alpha; // {0.0..n} multiplier of last weight change (momentum)
                         /* 0.0 no momentum
                           0.5 moderate momentum */

    static double transferFunction(double x) // sigmoid function is being used as the activation function
    {
        return 1 / (1 + exp(x));
    }

    static double transferFunctionDerivative(double x) // derivative of sigmoid
    {
        return transferFunction(x) * (1 - transferFunction(x));
    }

    static double randomWeight(void) // returns random double value
    {
        return rand() / double(RAND_MAX);
    }

    double sumDOW(const Layer &nextLayer)
    {
        double sum = 0.0;
        // Sum our contibution of the error at the nodes we feed

        for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
        {
            sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
        }

        return sum;
    }

public:
    vector<Connection> m_outputWeights; // vector of connection (weight and delta weight)

    neuron(unsigned numOutputs, unsigned myIndex)
    {
        for (unsigned c = 0; c < numOutputs; ++c) // c represents connections
        {
            m_outputWeights.push_back(Connection());        // appends a new element(connection) on to the output container with every iteration of loop
            m_outputWeights.back().weight = randomWeight(); // setting weight inside a connection randomly
        }

        m_myIndex = myIndex;
    }

    void setOutputVal(double val)
    {
        m_outputVal = val;
    }

    double getOutputVal(void)
    {
        return m_outputVal;
    }

    void feed_forward(Layer &prevLayer)
    {
        double sum = 0.0;

        // Sum the previous layer's output (which are our outputs)
        // Include the bias node's from the previous layer

        for (unsigned n = 0; n < prevLayer.size(); ++n)
        {
            sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
        }

        m_outputVal = neuron::transferFunction(sum);
    }

    void calcOutputGradients(double targetVal) // calculates gradient using target value and output value
    {
        double delta = targetVal - m_outputVal;
        m_gradient = delta * neuron::transferFunctionDerivative(m_outputVal);
    }

    void calcHiddenGradients(const Layer &nextLayer)
    {
        double dow = sumDOW(nextLayer);
        m_gradient = dow * neuron::transferFunctionDerivative(m_outputVal);
    }

    void updateInputWeights(Layer &prevLayer)
    {
        // The weights to be updated are in the connection container
        // in the neurons in the preceding layer
        for (unsigned n = 0; n < prevLayer.size(); ++n)
        {
            neuron &neuron = prevLayer[n];
            double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

            double newDeltaWeight =
                // Individual input, magnified by the gradient and train rate:
                eta *
                neuron.getOutputVal() * m_gradient *
                //  Also add momentum = a fraction of the previous delta weight
                +alpha * oldDeltaWeight;

            neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
            neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
        }
    }
};

double neuron::eta = 0.15;  // overall net learning rate {0.0..0.1}
double neuron::alpha = 0.5; // momentum, multiplier of last deltaWeight, {0.0..n}

// ****************** class neural net ***********************//
class neural_net
{
private:
    vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    static double m_recentAverageSmoothingFactor;

public:
    neural_net()
    {
        vector<unsigned> topology;
        // adding neurons to the layer
        topology.push_back(9);
        topology.push_back(5);
        topology.push_back(3);
        topology.push_back(3);
        topology.push_back(1);
        // topology used: 9,5,3,3,1

        unsigned numLayers = 3;
        for (unsigned i = 0; i < numLayers; i++)
        {
            m_layers.push_back(Layer());
            unsigned numOutputs = (i == topology.size() - 1) ? 0 : topology[i + 1]; // if last layer, 0 number of outputs

            for (unsigned j = 0; j <= topology[i]; j++)
            {
                m_layers.back().push_back(neuron(numOutputs, j));
                cout << "Made a neuron!" << endl;
            }

            m_layers.back().back().setOutputVal(1.0);
        }
    }

    void feed_forward(const vector<double> &inputVals)
    {
        assert(inputVals.size() == m_layers[0].size() - 1); // num of elements in inputVals = num of element in input layer
                                                            // input layer is a vector of neurons
                                                            //-1 is to account for the bias neuron

        // Assign (latch) the input values into the input neurons
        for (unsigned i = 0; i < inputVals.size(); ++i)
        {
            m_layers[0][i].setOutputVal(inputVals[i]);
        }
        // Forward propagate
        for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum)
        {
            Layer &prevLayer = m_layers[layerNum - 1];
            for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n)
            {
                m_layers[layerNum][n].feed_forward(prevLayer);
            }
        }
    }

    void back_propagation(const vector<double> &targetVals)
    {
        // Calculate the overall net error (RMS of output neuron errors)
        Layer &outputLayer = m_layers.back();
        m_error = 0.0;

        for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
        {
            double delta = targetVals[n] - outputLayer[n].getOutputVal();
            m_error += delta * delta;
        }
        m_error /= outputLayer.size() - 1; // get average error squared
        m_error = sqrt(m_error);           // RMS

        // implement a recent average measurement:
        m_recentAverageError =
            (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);

        // calculate output layer gradients
        for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
        {
            outputLayer[n].calcOutputGradients(targetVals[n]);
        }

        // calculate gradients on hidden layers
        for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
        {
            Layer &hiddenLayer = m_layers[layerNum];
            Layer &nextLayer = m_layers[layerNum + 1];

            for (unsigned n = 0; n < hiddenLayer.size(); ++n)
            {
                hiddenLayer[n].calcHiddenGradients(nextLayer);
            }
        }
        // For all layers fromoutput to first hidden layer
        // update connection weights

        for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
        {
            Layer &layer = m_layers[layerNum];
            Layer &prevLayer = m_layers[layerNum - 1];

            for (unsigned n = 0; n < layer.size() - 1; ++n)
            {
                layer[n].updateInputWeights(prevLayer);
            }
        }
    }

    void getResults(vector<double> &resultVals)
    {
        resultVals.clear();

        for (unsigned n = 0; n < m_layers.back().size() - 1; ++n)
        {
            resultVals.push_back(m_layers.back()[n].getOutputVal());
        }
    }

    double getRecentAverageError(void) const
    {
        return m_recentAverageError;
    }
};

double neural_net::m_recentAverageSmoothingFactor = 100.0;

class tictactoe
{
    vector<double> board; // keeps track of which player has placed a move at each poition
    char disp[9];         // only meant for display. shows the position number if no move placed there, or O or X if move has been placed

    neural_net myneural_net;

public:
    tictactoe()
    {
        for (int i = 0; i < 9; i++)
        {
            board.push_back(0); // initialized with 0 as 0 represents empty position
            disp[i] = i + 49;   // converting from int to char. fills up the board with nunmbers 1-9
        }
    }

    void display()
    {
        cout << "  " << disp[0] << "  |  " << disp[1] << "  |  " << disp[2] << "  " << endl;
        cout << "-----|-----|-----" << endl;
        cout << "  " << disp[3] << "  |  " << disp[4] << "  |  " << disp[5] << "  " << endl;
        cout << "-----|-----|-----" << endl;
        cout << "  " << disp[6] << "  |  " << disp[7] << "  |  " << disp[8] << "  " << endl;
    }

    vector<double> getBoardState()
    {
        return board;
    }

    void train()
    {
        int val;
        vector<double> inputvals, outputvals, targetvals, resultvals;
        fstream finvals("tic-tac-toe.txt");
        string ln;

        // now we have to read input and target value from the file
        for (int j = 0; j < 958; j++)
        {
            getline(finvals, ln);

            for (int l = 0, k = 0; l < 9; l++, k += 2) // k+=2 to skip the space
            {
                if (ln[k] == 'x')
                    inputvals.push_back(-1); // ai move
                else if (ln[k] == 'o')
                    inputvals.push_back(1); // users mov
                else
                    inputvals.push_back(0); // empty position
            }

            myneural_net.feed_forward(inputvals);
            myneural_net.getResults(resultvals);

            if (j < 626)
            {
                targetvals.push_back(1); // first 626 values of output file are positive so no need to read the file
                // cout << "target value: 1\n";
            }
            else
            {
                targetvals.push_back(0); // rest are 1
                // cout << "target value: 1\n";
            }

            myneural_net.back_propagation(targetvals);

            inputvals.clear();
            targetvals.clear();

            cout << "average error: " << myneural_net.getRecentAverageError() << endl;
        }
    }

    void move(double player, int opt) // places move at position mentioned. player == -1 for computer and player == 1 for user
    {
        char ch;
        if (player == -1) // AI
            ch = 'x';
        else if (player == 1) // human
            ch = 'o';

        switch (opt)
        {
        case 1:
            board[0] = player;
            disp[0] = ch;
            break;
        case 2:
            board[1] = player;
            disp[1] = ch;
            break;
        case 3:
            board[2] = player;
            disp[2] = ch;
            break;
        case 4:
            board[3] = player;
            disp[3] = ch;
            break;
        case 5:
            board[4] = player;
            disp[4] = ch;
            break;
        case 6:
            board[5] = player;
            disp[5] = ch;
            break;
        case 7:
            board[6] = player;
            disp[6] = ch;
            break;
        case 8:
            board[7] = player;
            disp[7] = ch;
            break;
        case 9:
            board[8] = player;
            disp[8] = ch;
            break;
        }
    }

    int istaken(int opt) // returns 0 if a certain position is empty.
    {
        switch (opt)
        {
        case 1:
            return board[0];
        case 2:
            return board[1];
            break;
        case 3:
            return board[2];
            break;
        case 4:
            return board[3];
            break;
        case 5:
            return board[4];
            break;
        case 6:
            return board[5];
            break;
        case 7:
            return board[6];
            break;
        case 8:
            return board[7];
            break;
        case 9:
            return board[8];
            break;
        }

        return 1; // for invalid option
    }

    void AImove()
    {

        vector<double> result;
        vector<double> tempBoard = getBoardState();
        vector<double> resultsarr; // result b/w 0 and 1 for each of the scenario. the higher the result, the higher probablity of winning

        for (int i = 0; i < 9; i++)
        {
            resultsarr.push_back(-1);
        }

        for (int i = 1; i < 10; i++)
        {
            if (!istaken(i))
            {
                tempBoard = getBoardState();
                tempBoard[i - 1] = -1; // making a move at position i-1. (tempboard has elements from 0-8 which is why i-1 is used)
                myneural_net.feed_forward(tempBoard);
                myneural_net.getResults(result);
                resultsarr[i - 1] = result[0]; // storing output from the ann in resultsarr
                tempBoard[i - 1] = 0;          // removing move from position i-1
            }
        }

        int maxIndex = 0; // now we have to find which input will give the highest output

        for (int i = 1; i < 9; i++)
        {
            if (resultsarr[i] > resultsarr[maxIndex])
            {
                maxIndex = i;
            }
        }

        move(-1, maxIndex + 1); // maxindex is from 0-8 while the function move expects a number from 1-9
    }

    void playermove()
    {
        int opt;
        while (1)
        {
            cout << "PLAYER: enter your option\n";
            cin >> opt;

            if (opt > 9 || opt < 1)
                cout << "enter valid option\n";
            else if (!istaken(opt))
            {
                move(1, opt); // places 1 at position opt
                break;
            }
            else
                cout << "choose an empty space\n";
        }
    }

    int iswon() // returns: 0 if game has not been won yet, -1 if computer wins, or 1 if user wins
    {
        // checking horisontal lines to see if they match
        if (board[0] == board[1] && board[1] == board[2] && board[1] != 0)
            return board[1];
        else if (board[3] == board[4] && board[4] == board[5] && board[4] != 0)
            return board[4];
        else if (board[6] == board[7] && board[7] == board[8] && board[6] != 0)
            return board[7];

        // checking vertical lines to see if they match
        else if (board[0] == board[6] && board[6] == board[3] && board[0] != 0)
            return board[0];
        else if (board[1] == board[4] && board[4] == board[7] && board[1] != 0)
            return board[1];
        else if (board[2] == board[5] && board[5] == board[8] && board[8] != 0)
            return board[2];

        // checking first diagonal
        else if (board[0] == board[4] && board[4] == board[8] && board[8] != 0)
            return board[0];

        // checking second diagonal
        else if (board[2] == board[4] && board[4] == board[6] && board[6] != 0)
            return board[2];

        return 0;
    }

    bool isdraw() // returns 1 if the game ends in a draw
    {
        if (!iswon()) //  can only be draw if no one has won the game yet
        {
            int count = 0;
            for (int i = 1; i <= 9; i++)
                if (istaken(i))
                    count++;

            if (count == 9) // if all positions are filled
                return 1;

            else
                return 0;
        }

        return 0;
    }

    void play()
    {

        train();
        cout << "\n\nCOMPUTER: X\nPLAYER: O\n\n";

        while (1)
        {
            display();
            playermove();

            cout << endl
                 << endl;

            if (iswon() || isdraw())
                break;

            display();
            AImove();

            cout << endl
                 << endl;

            if (iswon() || isdraw())
                break;
        }

        display();

        if (isdraw())
            cout << "GAME OVER. DRAW\n";
        else if (iswon() == -1)
            cout << "COMPUTER WINS THE GAME\n";
        else
            cout << "PLAYER WINS THE GAME\n";
    }
};

int main()
{
    srand(time(0));

    tictactoe game;

    game.play();

    system("pause");

    return 0;
}
