# Deblurring Convolutional Neural Network
Deep CNN implementation in Tensorflow for Deblurring text images

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Acknowldegements](#acknowledgements)


## Installation

Clone the project,

```sh
> git clone git@github.com:satwikkansal/deblurring_cnn.git
> cd deblurring_cnn
```

Install the dependencies,

```sh
> pip install -r requirements.txt
```

## Usage

Dataset can be downloaded by following the instructions [here](http://www.fit.vutbr.cz/~ihradis/CNN-Deblur/). Once downloaded, place the data accordingly in the `data/train` and `data/test` directory.

Begin the training with the following command,
```sh
> python train.py
```

The outputs are saved in the `output` directory.


## Contributing

All patches welcome!

## License

MIT License - see the [LICENSE](https://github.com/satwikkansal/readme_styles/blob/master/LICENSE) file for details


## Acknowledgements

- Reference: http://www.fit.vutbr.cz/~ihradis/pubs.php?file=%2Fpub%2F10922%2Fhradis15CNNdeblurring.pdf&id=10922

