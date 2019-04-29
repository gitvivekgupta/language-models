
# Sentence Generation with Trigram Language Modeling

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

- Perplexity on test set for __add-1 smoothing__ - 442.36

- Perplexity on test set for __simple interpolation__ - 79.80

- Sample sentences generated using __add-1 smoothing__:
    -  well , then i think , uh , got a bunch of lakes in our national
    -  i grow up with this five day waiting period is one of the upper peninsula ,
    -  we , we , uh , barbecue and tell me how to say in germany .
    -  and i felt real safe with their ten key or something like that ?
    -  raised it up in front of my boys .

- Sample sentences generated using __simple interpolation method__:
    - know , those things , i saw in the water ,
    - mean , i only had a pretty , you know , to , we kind of running around when you do n't really do n't know
    - he , uh , they all do , i have n't
    - uh , now i do about that
    - i go , and then you 've , we used to take their <UNK> <UNK> the , you know ,