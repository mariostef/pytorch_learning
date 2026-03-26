{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "id": "fY94u_PvibdH"
      },
      "outputs": [],
      "source": [
        "# For tips on running notebooks in Google Colab, see\n",
        "# https://docs.pytorch.org/tutorials/beginner/colab\n",
        "%matplotlib inline"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "gJyK4exaibdI"
      },
      "source": [
        "Neural Networks\n",
        "===============\n",
        "\n",
        "Neural networks can be constructed using the `torch.nn` package.\n",
        "\n",
        "Now that you had a glimpse of `autograd`, `nn` depends on `autograd` to\n",
        "define models and differentiate them. An `nn.Module` contains layers,\n",
        "and a method `forward(input)` that returns the `output`.\n",
        "\n",
        "For example, look at this network that classifies digit images:\n",
        "\n",
        "![convnet](https://pytorch.org/tutorials/_static/img/mnist.png)\n",
        "\n",
        "It is a simple feed-forward network. It takes the input, feeds it\n",
        "through several layers one after the other, and then finally gives the\n",
        "output.\n",
        "\n",
        "A typical training procedure for a neural network is as follows:\n",
        "\n",
        "-   Define the neural network that has some learnable parameters (or\n",
        "    weights)\n",
        "-   Iterate over a dataset of inputs\n",
        "-   Process input through the network\n",
        "-   Compute the loss (how far is the output from being correct)\n",
        "-   Propagate gradients back into the network's parameters\n",
        "-   Update the weights of the network, typically using a simple update\n",
        "    rule: `weight = weight - learning_rate * gradient`\n",
        "\n",
        "Define the network\n",
        "------------------\n",
        "\n",
        "Let's define this network:\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3Mtfs53LibdK",
        "outputId": "73d58ab8-595f-4a28-9a27-d442301f4d23"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Net(\n",
            "  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))\n",
            "  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))\n",
            "  (fc1): Linear(in_features=400, out_features=120, bias=True)\n",
            "  (fc2): Linear(in_features=120, out_features=84, bias=True)\n",
            "  (fc3): Linear(in_features=84, out_features=10, bias=True)\n",
            ")\n"
          ]
        }
      ],
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "\n",
        "\n",
        "class Net(nn.Module):\n",
        "\n",
        "    def __init__(self):\n",
        "        super(Net, self).__init__()  #Γραμμή 1: Η κλάση μας κληρονομεί από nn.Module\n",
        "                                     #Γραμμή 2: Constructor (τρέχει όταν κάνεις net = Net())\n",
        "                                     #Γραμμή 3: Καλεί τον parent constructor (ΠΑΝΤΑ χρειάζεται!)\n",
        "        # 1 input image channel, 6 output channels, 5x5 square convolution\n",
        "        # kernel\n",
        "        self.conv1 = nn.Conv2d(1, 6, 5)\n",
        "        self.conv2 = nn.Conv2d(6, 16, 5)\n",
        "        # an affine operation: y = Wx + b\n",
        "        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension\n",
        "        self.fc2 = nn.Linear(120, 84)\n",
        "        self.fc3 = nn.Linear(84, 10) #output 10 logw 10 classes\n",
        "\n",
        "    def forward(self, input):\n",
        "        # Convolution layer C1: 1 input image channel, 6 output channels,\n",
        "        # 5x5 square convolution, it uses RELU activation function, and\n",
        "        # outputs a Tensor with size (N, 6, 28, 28), where N is the size of the batch\n",
        "        c1 = F.relu(self.conv1(input))\n",
        "        # Subsampling layer S2: 2x2 grid, purely functional,\n",
        "        # this layer does not have any parameter, and outputs a (N, 6, 14, 14) Tensor\n",
        "        s2 = F.max_pool2d(c1, (2, 2))\n",
        "        # Convolution layer C3: 6 input channels, 16 output channels,\n",
        "        # 5x5 square convolution, it uses RELU activation function, and\n",
        "        # outputs a (N, 16, 10, 10) Tensor\n",
        "        c3 = F.relu(self.conv2(s2))\n",
        "        # Subsampling layer S4: 2x2 grid, purely functional,\n",
        "        # this layer does not have any parameter, and outputs a (N, 16, 5, 5) Tensor\n",
        "        s4 = F.max_pool2d(c3, 2)\n",
        "        # Flatten operation: purely functional, outputs a (N, 400) Tensor\n",
        "        s4 = torch.flatten(s4, 1)\n",
        "        # Fully connected layer F5: (N, 400) Tensor input,\n",
        "        # and outputs a (N, 120) Tensor, it uses RELU activation function\n",
        "        f5 = F.relu(self.fc1(s4))\n",
        "        # Fully connected layer F6: (N, 120) Tensor input,\n",
        "        # and outputs a (N, 84) Tensor, it uses RELU activation function\n",
        "        f6 = F.relu(self.fc2(f5))\n",
        "        # Fully connected layer OUTPUT: (N, 84) Tensor input, and\n",
        "        # outputs a (N, 10) Tensor\n",
        "        output = self.fc3(f6)\n",
        "        return output\n",
        "\n",
        "\n",
        "net = Net()\n",
        "print(net)"
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "wPd8pa4mkBGV"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "1️⃣ Δομή nn.Module (ΠΑΝΤΑ η ίδια!)Κάθε PyTorch μοντέλο έχει αυτή τη δομή:pythonclass MyModel(nn.Module):\n",
        "    def __init__(self):\n",
        "        super().__init__()\n",
        "        # ΕΔΩ: Δηλώνω τα layers\n",
        "        \n",
        "    def forward(self, x):\n",
        "        # ΕΔΩ: Ορίζω τη ροή δεδομένων\n",
        "        return xΓιατί;\n",
        "\n",
        "nn.Module = η βάση για ΟΛΑ τα μοντέλα\n",
        "__init__ = δηλώνω ΤΙ κομμάτια έχει το δίκτυο\n",
        "forward = ορίζω ΠΩΣ συνδέονται τα κομμάτια.Για επεξεργασία φωνής:\n",
        "self.embed = nn.Embedding(\n",
        "    num_embeddings=10000,  # Πόσες λέξεις ξέρει\n",
        "    embedding_dim=50       # Διαστάσεις κάθε λέξης\n",
        ")\n",
        "\n",
        "B. nn.Linear (πολλαπλασιασμός πινάκων)\n",
        "pythonself.fc = nn.Linear(\n",
        "    in_features=50,   # Πόσα inputs\n",
        "    out_features=128  # Πόσα outputs\n",
        ")\n",
        "```\n",
        "\n",
        "**Τι κάνει;**  \n",
        "Μαθηματικά: `y = x @ W + b`\n",
        "```\n",
        "[50 αριθμοί] → [Linear] → [128 αριθμοί]\n",
        "\n",
        "self.relu = nn.ReLU()\n",
        "```\n",
        "\n",
        "**Τι κάνει;**  \n",
        "`ReLU(x) = max(0, x)` → Αρνητικοί γίνονται 0\n",
        "```\n",
        "Input:  [-2, -1, 0, 1, 2]\n",
        "Output: [ 0,  0, 0, 1, 2]"
      ],
      "metadata": {
        "id": "_pjraVUQnc87"
      }
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "xawHp17VibdL"
      },
      "source": [
        "You just have to define the `forward` function, and the `backward`\n",
        "function (where gradients are computed) is automatically defined for you\n",
        "using `autograd`. You can use any of the Tensor operations in the\n",
        "`forward` function.\n",
        "**Γράφω μόνο το forward.Το backword γίνεται αυτόματα.**\n",
        "\n",
        "The learnable parameters of a model are returned by `net.parameters()`\n",
        "**net.parameters() → όλα τα βάρη του μοντέλου**\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QhY4v09dibdL",
        "outputId": "de74e7d8-0cd6-4b01-ec6a-b403d5dca370"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "10\n",
            "torch.Size([6, 1, 5, 5])\n"
          ]
        }
      ],
      "source": [
        "params = list(net.parameters())\n",
        "print(len(params))\n",
        "print(params[0].size())  # conv1's .weight"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "zkEnVvnMibdM"
      },
      "source": [
        "Let\\'s try a random 32x32 input. Note: expected input size of this net\n",
        "(LeNet) is 32x32. To use this net on the MNIST dataset, please resize\n",
        "the images from the dataset to 32x32.\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "tTWw44N3o32k"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "h_afLFSmibdM",
        "outputId": "1ea20fa6-fc46-4d2c-8b95-3f931c597e33"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "tensor([[ 0.0628, -0.0190, -0.1290,  0.1062, -0.0867,  0.0117, -0.0343, -0.0005,\n",
            "          0.1275,  0.0224]], grad_fn=<AddmmBackward0>)\n"
          ]
        }
      ],
      "source": [
        "input = torch.randn(1, 1, 32, 32) #batch size,channels,pixels\n",
        "out = net(input)\n",
        "print(out)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "zocDgj6FibdM"
      },
      "source": [
        "Zero the gradient buffers of all parameters and backprops with random\n",
        "gradients:\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 6,
      "metadata": {
        "id": "jTMgxvoUibdN"
      },
      "outputs": [],
      "source": [
        "net.zero_grad()\n",
        "out.backward(torch.randn(1, 10))#ειπαμε δεν γραφω κωδικα για backward αλλα πρεπει  να το καλεσω εγω\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "OLk-Kk5IibdN"
      },
      "source": [
        "<div style=\"background-color: #54c7ec; color: #fff; font-weight: 700; padding-left: 10px; padding-top: 5px; padding-bottom: 5px\"><strong>NOTE:</strong></div>\n",
        "\n",
        "<div style=\"background-color: #f3f4f7; padding-left: 10px; padding-top: 10px; padding-bottom: 10px; padding-right: 10px\">\n",
        "\n",
        "**<p><code>torch.nn</code> only supports mini-batches.** The entire <code>torch.nn</code>package only supports inputs that are a mini-batch of samples, and nota single sample.For example, <code>nn.Conv2d</code> will take in a 4D Tensor of<code>nSamples x nChannels x Height x Width</code>.If you have a single sample, just use <code>input.unsqueeze(0)</code> to adda fake batch dimension.</p>\n",
        "\n",
        "</div>\n",
        "\n",
        "Before proceeding further, let\\'s recap all the classes you've seen so\n",
        "far.\n",
        "\n",
        "**Recap:**\n",
        "\n",
        ":   -   `torch.Tensor` - A *multi-dimensional array* with support for\n",
        "        autograd operations like `backward()`. Also *holds the gradient*\n",
        "        w.r.t. the tensor.\n",
        "    -   `nn.Module` - Neural network module. *Convenient way of\n",
        "        encapsulating parameters*, with helpers for moving them to GPU,\n",
        "        exporting, loading, etc.\n",
        "    -   `nn.Parameter` - A kind of Tensor, that is *automatically\n",
        "        registered as a parameter when assigned as an attribute to a*\n",
        "        `Module`.\n",
        "    -   `autograd.Function` - Implements *forward and backward\n",
        "        definitions of an autograd operation*. Every `Tensor` operation\n",
        "        creates at least a single `Function` node that connects to\n",
        "        functions that created a `Tensor` and *encodes its history*.\n",
        "\n",
        "**At this point, we covered:**\n",
        "\n",
        ":   -   Defining a neural network\n",
        "    -   Processing inputs and calling backward\n",
        "\n",
        "**Still Left:**\n",
        "\n",
        ":   -   Computing the loss\n",
        "    -   Updating the weights of the network\n",
        "\n",
        "Loss Function\n",
        "=============\n",
        "\n",
        "A loss function takes the (output, target) pair of inputs, and computes\n",
        "a value that estimates how far away the output is from the target.\n",
        "\n",
        "There are several different [loss\n",
        "functions](https://pytorch.org/docs/nn.html#loss-functions) under the nn\n",
        "package . A simple loss is: `nn.MSELoss` which computes the mean-squared\n",
        "error between the output and the target.\n",
        "\n",
        "For example:\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 7,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "O_DrtMXYibdN",
        "outputId": "a2b46c76-ad2c-4e38-c8a5-04e8cd9e47a4"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "tensor(1.1879, grad_fn=<MseLossBackward0>)\n"
          ]
        }
      ],
      "source": [
        "output = net(input)\n",
        "target = torch.randn(10)  # a dummy target, for example\n",
        "target = target.view(1, -1)  # make it the same shape as output\n",
        "criterion = nn.MSELoss() #loss function για regression\n",
        "\n",
        "loss = criterion(output, target)\n",
        "print(loss)"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Loss Function\n",
        "- Μετράει πόσο λάθος είναι το μοντέλο\n",
        "- Input: (prediction, target)\n",
        "- Output: Ένας αριθμός (μικρότερος = καλύτερα!)\n",
        "\n",
        "## Για την άσκηση:\n",
        "- Binary: BCEWithLogitsLoss()\n",
        "- Multi-class: CrossEntropyLoss()"
      ],
      "metadata": {
        "id": "Huva7lsKrUE2"
      }
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "aYojLNFHibdO"
      },
      "source": [
        "Now, if you follow `loss` in the backward direction, using its\n",
        "`.grad_fn` attribute, you will see a graph of computations that looks\n",
        "like this:\n",
        "\n",
        "Δειχνει πως υπολογιστηκε το Loss.Ολο το computation graph\n",
        "\n",
        "``` {.sh}\n",
        "input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d\n",
        "      -> flatten -> linear -> relu -> linear -> relu -> linear\n",
        "      -> MSELoss\n",
        "      -> loss\n",
        "```\n",
        "\n",
        "So, when we call `loss.backward()`, the whole graph is differentiated\n",
        "w.r.t. the neural net parameters, and all Tensors in the graph that have\n",
        "`requires_grad=True` will have their `.grad` Tensor accumulated with the\n",
        "gradient.\n",
        "\n",
        "For illustration, let us follow a few steps backward:\n"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Computational Graph\n",
        "- Το PyTorch κρατάει την \"ιστορία\" των πράξεων\n",
        "- loss.backward() → Διαφορίζει όλο το graph\n",
        "- Υπολογίζει dL/dW για ΚΑΘΕ layer.\n",
        "Αυτές είναιοι ευθύνες.\n",
        "- Αποθηκεύει στο .grad attribute\n",
        "\n",
        "## Chain of computation:\n",
        "input → layers → loss\n",
        "loss → backward → gradients για όλα!"
      ],
      "metadata": {
        "id": "XJoZiljUr85J"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 8,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "d1SPM6z3ibdO",
        "outputId": "360272ed-2142-4a02-822c-662e72bb2dbd"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<MseLossBackward0 object at 0x7f23d3cd80a0>\n",
            "<AddmmBackward0 object at 0x7f23f96da9b0>\n",
            "<AccumulateGrad object at 0x7f23d3cd80a0>\n"
          ]
        }
      ],
      "source": [
        "print(loss.grad_fn)  # MSELoss\n",
        "print(loss.grad_fn.next_functions[0][0])  # Linear\n",
        "print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "XrgA1ltzibdP"
      },
      "source": [
        "Backprop\n",
        "========\n",
        "\n",
        "To backpropagate the error all we have to do is to `loss.backward()`.\n",
        "**You need to clear the existing gradients though, else gradients will be\n",
        "accumulated to existing gradients.**\n",
        "\n",
        "Now we shall call `loss.backward()`, and have a look at conv1\\'s bias\n",
        "gradients before and after the backward.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 9,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "T4JylDvkibdP",
        "outputId": "629ec58e-8a2c-4c7a-9250-8e8bf15f52d4"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "conv1.bias.grad before backward\n",
            "None\n",
            "conv1.bias.grad after backward\n",
            "tensor([-0.0016,  0.0020, -0.0114, -0.0148,  0.0117, -0.0036])\n"
          ]
        }
      ],
      "source": [
        "net.zero_grad()     # zeroes the gradient buffers of all parameters\n",
        "\n",
        "print('conv1.bias.grad before backward')\n",
        "print(net.conv1.bias.grad)\n",
        "\n",
        "loss.backward()\n",
        "\n",
        "print('conv1.bias.grad after backward')\n",
        "print(net.conv1.bias.grad)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "pyV5yiB7ibdP"
      },
      "source": [
        "Now, we have seen how to use loss functions.\n",
        "\n",
        "**Read Later:**\n",
        "\n",
        "> The neural network package contains various modules and loss functions\n",
        "> that form the building blocks of deep neural networks. A full list\n",
        "> with documentation is [here](https://pytorch.org/docs/nn).\n",
        "\n",
        "**The only thing left to learn is:**\n",
        "\n",
        "> -   Updating the weights of the network\n",
        "\n",
        "Update the weights\n",
        "==================\n",
        "\n",
        "The simplest update rule used in practice is the Stochastic Gradient\n",
        "Descent (SGD):\n",
        "\n",
        "``` {.python}\n",
        "weight = weight - learning_rate * gradient\n",
        "```\n",
        "\n",
        "We can implement this using simple Python code:\n",
        "\n",
        "``` {.python}\n",
        "learning_rate = 0.01\n",
        "for f in net.parameters():\n",
        "    f.data.sub_(f.grad.data * learning_rate)\n",
        "```\n",
        "\n",
        "However, as you use neural networks, you want to use various different\n",
        "update rules such as SGD, Nesterov-SGD, Adam, RMSProp, etc.**Αυτά ουσιαστικά το κάνουν το ίδιο κατευθείαν για όλες τις παραμέτρους χωρίς for. **To enable\n",
        "this, we built a small package: `torch.optim` that implements all these\n",
        "methods. Using it is very simple:\n",
        "\n",
        "``` {.python}\n",
        "import torch.optim as optim\n",
        "\n",
        "# create your optimizer\n",
        "optimizer = optim.SGD(net.parameters(), lr=0.01)\n",
        "\n",
        "# in your training loop:\n",
        "optimizer.zero_grad()   # zero the gradient buffers\n",
        "output = net(input)\n",
        "loss = criterion(output, target)\n",
        "loss.backward()\n",
        "optimizer.step()    # Does the update\n",
        "```\n"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [],
      "metadata": {
        "id": "dCGbwEYg8v7c"
      }
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "97B7Hi7mibdQ"
      },
      "source": [
        "<div style=\"background-color: #54c7ec; color: #fff; font-weight: 700; padding-left: 10px; padding-top: 5px; padding-bottom: 5px\"><strong>NOTE:</strong></div>\n",
        "\n",
        "<div style=\"background-color: #f3f4f7; padding-left: 10px; padding-top: 10px; padding-bottom: 10px; padding-right: 10px\">\n",
        "\n",
        "<p>Observe how gradient buffers had to be manually set to zero using<code>optimizer.zero_grad()</code>. This is because gradients are accumulatedas explained in the <a href=\"\">Backprop</a> section.</p>\n",
        "\n",
        "</div>\n",
        "\n"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.10.12"
    },
    "colab": {
      "provenance": []
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
