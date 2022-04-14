
data = ["the storyline is weak and the acting is good", "i've seen this show many times and still enjoy it", 'i can say we have seen a similar storyline before in a much better films']
from pysentimiento import create_analyzer
analyzer = create_analyzer(task="sentiment", lang="es")

a = analyzer.predict("the storyline is weak and the acting is good")
# returns AnalyzerOutput(output=POS, probas={POS: 0.998, NEG: 0.002, NEU: 0.000})
b = analyzer.predict("i've seen this show many times and still enjoy it")
# returns AnalyzerOutput(output=NEG, probas={NEG: 0.999, POS: 0.001, NEU: 0.000})
c = analyzer.predict('i can say we have seen a similar storyline before in a much better films')
d = analyzer.predict('the storyline is about a young woman who is killed by a young woman who')


# returns AnalyzerOutput(output=NEU, probas={NEU: 0.993, NEG: 0.005, POS: 0.002})
print('get')