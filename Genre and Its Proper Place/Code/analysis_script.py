""" This module contains some scripts that can be used
to analyze a text.
"""
import os
import nltk
import nltk.tag.stanford as stanford
from nltk.tokenize.punkt import PunktSentenceTokenizer as SentenceTokenizer
from pygeocoder import Geocoder
from django.template import Template, Context
from django.conf import settings
from textblob.en.sentiments import PatternAnalyzer
import matplotlib.pyplot as plt
import numpy as np
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers.english import stem_word
from sumy.utils import get_stop_words

settings.configure()

### need to set the JAVAHOME variable to use stanford parser
os.environ['JAVAHOME'] = "PUT THE PATH TO YOUR JAVA FILE HERE"
nltk.internals.config_java("PUT THE PATH TO YOUR JAVA FILE HERE")

# CONSTANTS

NERTAGGER_LOC = 'PUT THE PATH TO YOUR STANDFORD NER TAGGER HERE'
NERTAGGER_TRAINING_DATA = 'PUT THE PATH TO YOUR TRAINING DATA HERE'
kml_template = (
"""
<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
 <Document>
 {% for place in places %}
  <Placemark>
    <name> {{ place.name }} </name>
    <description> {{ place.name }} </description>
    <Point>
      <coordinates> {{ place.lon }},{{ place.lat }},0</coordinates>
    </Point>
  </Placemark>
  {% endfor %}
 </Document>
</kml>
"""
)

class Place:

    def __init__(self, name, lat, lon):

        self.name = name
        self.lat = lat
        self.lon = lon

## Function meant to change "New" "England" to "New England"
    
class BookReader:

    def __init__(self, corpus, *args, **kwargs):
        """ corpus: path to the file containing the
        the text of the corpus to be studied.
        """
        corpus_file = open(corpus)
        self.corpus = corpus_file.read()
        corpus_file.close()

    @property
    def sentences(self):
        try:
            return self.sentences_list

        except(AttributeError):
            sentence_tokenizer = SentenceTokenizer()
            self.sentences_list = sentence_tokenizer.tokenize(self.corpus)
            return self.sentences_list

    @property
    def named_entities(self):

        try:
            # return named entity tokens:
            # [('word', 'PERSON/ORGANIZATION/etc...'), ... ]
            return self.named_entity_tokens

        except(AttributeError):
            # create named entity list
            st = stanford.NERTagger(NERTAGGER_TRAINING_DATA, NERTAGGER_LOC)
            rough_named_entities = st.tag(
                # grammatical symbols cause an unknown error stopping the tagger, so I remove them
                nltk.tokenize.word_tokenize(self.corpus.replace('.', "").replace('?','').replace('!','')))
            self.named_entity_tokens_raw = filter(lambda x: x[1] != 'O', rough_named_entities)
            self.named_entity_tokens = set(self.named_entity_tokens_raw)
            return self.named_entity_tokens

    @property
    def named_entities_raw(self):

        try:
            return self.named_entity_tokens_raw
        except:
            self.named_entities
            return self.named_entity_tokens_raw

    @property
    def locations(self):

        try:
            # return location list
            return self.location_list
        
        except(AttributeError):
            # generate the location list
            location_list = []
            for token in self.named_entities:
                if token[1] == 'LOCATION' :
                    location_list.append(token[0])
            self.location_list = location_list
            return self.location_list

    def locations_kml(self):
        # create a kml file of placemarkers for google earth
        # from the locations in the book
        template = Template(kml_template)
        places = []
        i = 0
        for location in set(self.locations):
            i += 1
            print i
            try:
                results = Geocoder.geocode(location)
                if results.count <= 1:
                    coordinates = results[0].coordinates
                    lat = coordinates[0]
                    lon = coordinates[1]
                    places.append(Place(location, lat, lon))
            except:
                pass
        context = Context({'places': places})
        output_text = template.render(context)
        output = open('locations.kml', 'w')
        output.write(output_text)
        output.close()

    @property
    def sentiment_trends(self):
        # sentiment_trends_tup =
        #     (sentiment_trends_list_sentences,
        #      sentiment_trends_list_chunks)
        try:
            return self.sentiment_trends_tup
        
        except:
            sent_chunk_size = 200
            step_size = 40
            sentiment_trends_list_sentences = []
            sentiment_trends_list_chunks = []
            # generate a list of sentiments of sections
            pattern_analyzer = PatternAnalyzer()
            for i in xrange(0, len(self.sentences), step_size):
                print i
                sentiment_trends_list_sentences.append(
                    pattern_analyzer.analyze(
                        self.sentences[i]))
                if i + sent_chunk_size <= len(self.sentences) :
                    sentiment_trends_list_chunks.append(
                        pattern_analyzer.analyze(
                            "".join(self.sentences[i:i+sent_chunk_size])))
            self.sentiment_trends_tup = (sentiment_trends_list_sentences,
                                         sentiment_trends_list_chunks)
            return self.sentiment_trends_tup

    def sentiment_trends_graph(self):
        # graph sentiment over time
        plt.figure(1)
        plt.subplot(211)
        plt.plot(map(lambda x:x.polarity, self.sentiment_trends[1]))
        plt.ylabel("polarity")
        plt.ylim(-0.3, 0.3)

        plt.subplot(212)
        plt.plot(map(lambda x:x.subjectivity, self.sentiment_trends[1]))
        plt.ylabel("subjectivity")
        plt.xlabel("distance through text")
        plt.ylim(0,1)

        plt.show()

        # histogram of emotional content
        plt.figure(2)

        plt.subplot(211)
        n, bins, patches = plt.hist(
            map(lambda x:x.polarity,
                self.sentiment_trends[0]),
            50,
            facecolor='b')
        plt.xlabel('positive - negative emotion')
        plt.ylabel('number of sentences')
        plt.title('Sentiment/Subjectivity Histograms')
        plt.grid(True)

        plt.subplot(212)
        n, bins, patches = plt.hist(
            map(lambda x:x.subjectivity,
                self.sentiment_trends[0]),
            50,
            facecolor='g')
        plt.xlabel('subjectivity')
        plt.ylabel('number of sentences')
        plt.grid(True)
        
        plt.show()


    def summary(self, int1, int2):
        # int1, int2 are the places between which to look for
        # the summary to be taken (slicing the corpus as a string)
        parser = PlaintextParser(self.corpus[int1:int2], Tokenizer("english"))
        summarizer = LsaSummarizer(stem_word)
        summarizer.stop_words = get_stop_words("english")
        self.summary_text = " ".join(
            map(lambda x:x._text,
                summarizer(parser.document, 20)))
        return self.summary_text

    @property
    def named_entity_counts(self):
        try:
            return self.named_entity_counts_list
        
        except:
            self.named_entity_counts_list = []
            named_entities = self.named_entities
            named_entities_raw = self.named_entities_raw
            for entity in named_entities:
                self.named_entity_counts_list.append((entity, named_entities_raw.count(entity)))
            return self.named_entity_counts_list

    @property
    def character_counts(self):
        try:
            return self.character_counts_list

        except:
            self.character_counts_list = filter(lambda x:x[0][1] == 'PERSON',
                                                self.named_entity_counts)
            return self.character_counts_list

    @property
    def place_counts(self):
        try:
            return self.place_counts_list
        except:
            self.place_counts_list = filter(lambda x:x[0][1] == 'LOCATION',
                                            self.named_entity_counts)
            return self.place_counts_list

    @property
    def organization_counts(self):
        try:
            return self.organization_counts_list
        except:
            self.organization_counts_list = filter(lambda x:x[0][1] == 'ORGANIZATION',
                                                   self.named_entity_counts)
            return self.organization_counts_list
            
        
