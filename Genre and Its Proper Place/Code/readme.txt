You will need to download several python libraries and a version of the Stanford NER Tagger in order to get this script to work. Then type the lines:

book_reader = BookReader('Path to text to analyze")
book_reader.locations_kml()

into the interpreter. This will output the desired kml file which then can be opened in Google Earth.