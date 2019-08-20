import BlazeFace

shape = (512, 512)


def main():
    # build training pipeline
    inputs, phrase_train, predict2, predict3, predict4, predict5 = BlazeFace.build_prediction_convs(shape)


if __name__ == '__main__':
    main()
