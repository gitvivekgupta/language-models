
# Sentence Generation

Implemented trigram language model with unknown word handling (replace words of frequency less than 5 as UNK). The code also handles different smoothing techniques like add-1 smoothing and simple interpolation smoothing. It then computes the perplexity on the test on both the smoothing methods, so as to compare and analyze as to which method is better.

Once the language model has been generated using both methods, we try to generate sentences using the Shannon Method.

## Dataset

The Switchboard Corpus(http://compprag.christopherpotts.net/swda.html) will be used for training and testing the model. The Switchboard Corpus is a telephone speech corpus containing conversations between people. It has the fields namely, filename, basename, conversationId, number of lines in conversation, act tags, conversing person ID, utterance indices, subutterance and the text (conversation) itself.

There are 3 files namely, testset.csv, trainset.csv and devset.csv. Here, trainset.csv has 70% of the total data, and devset.csv and testset.csv each has 15%.

## Steps taken

1. Extracted the text field from each line of each csv file.
2. Remove all the unnecessary symbols leaving behind meaningful words.
3. Lower cased all the words.
4. Kept a count of each word appearing in the corpus which helps generalize which words appear less than 5 times as "UNK".
5. Calculate the probability using add-1 and simple interpolation and then compute perplexity.
6. Generated 20 sentences from each of the model using the Shannon method.

## Results

- Perplexity on test set for __add-1 smoothing__ - 442.31

- Perplexity on test set for __simple interpolation__ - 79.80

- Sample sentences generated using __add-1 smoothing__:
    -  <s> well , actually i like seeing other countries to get it across nicaragua , </s>
    -  <s> no </s>
    -  <s> yeah . </s>
    -  <s> so i was growing up that i see , </s>
    -  <s> yeah . </s>

![Screenshot 2019-04-29 at 4 27 32 PM](https://user-images.githubusercontent.com/17769945/56891944-bf17e480-6a9b-11e9-9095-5f25f5b01280.png)



- Sample sentences generated using __simple interpolation method__:

![Screenshot 2019-04-29 at 4 28 26 PM](https://user-images.githubusercontent.com/17769945/56891995-dbb41c80-6a9b-11e9-9a1d-30657250cd9a.png)
