

### Swiss Procurement -- Jan (2m)

### Problem description -- Jan (1m)

# Aim -- Alex (1m)
- Wie gezeigt ist der Prozess nicht optimal
- Jetztiger Prozess legt man sich auf CPVs fest und erhält so eine limitierte Auswahl
- **Gefahr**: Es werden Ausschreibungen übersehen, wenn man seine Filter nicht gut genung setzt *oder* man muss zu viele Ausschreibungen durchkämmen. Ausserdem kostet der ganze Prozess Zeit und Aufwand
- **Ideal**: Diesen Prozess weiter automatisieren.
	- Dafür ist jedoch Wissen über die einzelene Auftraggeber, die Ausschreigungen und Anbieter nötig
- Wo kriegen wir diese Wissen her und wie implementieren wir es in einen automatisierten Prozess?
	- Möglichkeit die von der FDN gesammelten Beschaffungsdaten als das "Wissen" zu verwenden
	- Durch ein "datengetriebenes" Vorgehen aus den Daten lernen und den Prozess automatisieren
- Und so Business Value zu liefern: Gleich gute oder bessere Ausschreigungsauswahl, schnellere Prozesse, weniger Aufwand, mehr Markttransparenz wenn alle passenden Ausschreibungen ersichtlich werden, Passende Anbiert für die jeweiligen Ausschreibungen
- Ziel der Arbeit: Ist mit der vorhandenen Datengrundlage eine Aussage über interessante Ausschreibungen möglich? --> Erste Implementation



# Methodology -- Alex (1m)

- Wie wir vorgegangen sind spiegelt sich gleich im Rest der Präsentation wieder:
- Zuerst eine Übersicht über alle Daten verschafft, einige desktiptive Analysen duchgeführt und die verschiedenen Abhängigkeiten studiert
- Dann in die Welt des statistischen Lernens eingearbeitet um eine Idee zu erhalten, wie man aus Daten eine Entscheidungsmethodik gestalten kann
- Von bestehenden Machine Learning Algorithmen den Passenden ausgewählt.
- Danach haben wir uns für einen Lerndatensatz entschieden, zu welchem wir sukzessive einzelne Attribute zuerst bereinigt und dann in den Lernprozess aufgenommen haben



# Tools
- Umsetzung in der Programmiersprache Python, mit Libraries Pandas & Numpy für das Datahandling und SciKit Learn für die nötigen Machine Learning Instrumente (open source und community driven).
- Mit Jupyter Notebook ist ein Werkzeug, in welchem Daten als Code Snippets in Zellen aneinander gekettet werden. So ist der Werdegang des Codes später einfach nachvollzieh- und reproduzierbar
- Daten genauer kennenlernen: Jan

### Data presentation -- Jan (3m)

# Machine Learning (ML) – Terminology
- Kurz einige Begriffe klären
- Was ist eigentlich ML? SAS Institutn erklärt es ziemlich kurz und präzise:
- "*Machine learning* is a method of data analysis that automates analytical model building."
- "[...] idea that systems can learn from data, identify patterns and make decisions with minimal human intervention."
- ML wird in zwei (vier) grosse Kategorien geteilt
- Beim Supervised ML versucht das System den Zusammenhang von Input (*predictors*) und Outputvariablen (*responses*) zu verstehen und diesen in einem mathematischen Modell festzuhalten. Um diese Verhältnis zu "lernen" benötigt man beim Supervised Learning angaben über die "Richtigkeit" von Zusammenhängen. Die Daten müsssen also mit einer Art Label versehen sein.
- Beim Unsupervised learning sind die Daten nicht nicht gelabelt, es fehlt also ein offensichtlicher Output. Hier versucht man eher die Daten zu erkunden und zu verstehen, indem man z.B. Cluster bildet oder die Dimensionen der Daten zu verringern um daraus explorative Schlüsse zu ziehen



# ML – Situational Assessment
- Um nun eine Kategorie ...
- Unser grosses Problem war, dass wir für Supervised learning nicht die nötigen Responses zur Verfügung hatten: Wir können aus unsern Daten zwar herauslesen, wer eine Auschreigung gewonnen hat, nicht aber für welche Ausschreibungen sich ein Anbieter generell interessiert, oder für sich die Anbieter nicht interessieren!
- Im Pool aller Ausschreibungen befinden sich folglich Interessante und nicht interessante Objekte
- Da uns also die Unterscheidung in negative und positive Responses fehlt, haben wir uns für unsupervised learning entschieden.
- Wir wollten aufgrund vergangener gewonnener Ausschreibungen verschieden Cluster bilden, zu denen dann für neue Ausschreibungen eine gewissen Ähnichkeitsnähe im Mehrdimensionalen Raum messen wollten. Wenn die neue Ausschreibung in diesem Raum nahe des "interessnaten" Clusters angesiedelt ist, können wir davon ausgehen, dass sich der Anbieter dafür interessieren könnte.
- **Beim Testen diesen Ansatzes ist uns jedoch aufgefallen, das uns bei diesem Vorgehen ein klare Messbarkeit des Erfolges fehlte.**
- **Die Überprüfung der Implementation könnten wir entweder nur mit viel Domänenwissen des jeweiligen Abieters überprüfen [XXX Beispiel einfügen]**
- **ODER wir müssten die erfolgreich gewonnen Ausschreibungen zu Testzwecken benutzen, was jedoch wiederum die Zusammensetzung des Clusters beeinträchtigen könnte**
- **Zudem wäre es wohl schwierig geworden eine Clusterthreshold festzusetzen, die sich wohl in den unterschiedlichen Businessdomänen sehr unterscheidelich zusammensetzen würde**
- **[XXX Check all the bold again]**
- Daher sind wir wieder zurück auf Feld eins und kamen dann auf die Idee doch dem Supervised Ansatz zu folgen: Unsere Idee lag darin, dass wir die gewonnenen Ausschreibungen als Positive responses behandeln und alle anderen als negative Responses. Ausschreibungen, die für den Anbieter ebenfalls positiv wären, werden so zwar als negativ trainiert, im zweifelsfall jedoch (eigentlich fälschlicherweise) als positiv klassifiziert.


# ML – Algorithm Choice
- Bei der Wahl des ML Algo gilt das sogenannt "No Free Lunch" Theorem: Es gibt keinen idealen Algo für jedes Problem. Grösse und Struktur des Datensets haben einen Einfluss, plus viele weitere Faktoren
- Zur Wahl des eines bestimmten ML Algorithmus sind wir ungefähr der Roadmap von SAS Institute Inc. und der Empfehlung von SciKit Learn gefolt. Das Vorgehen haben wir in dieser vereinfachten Darstellung zusammengefasst.
- RF weil schnelle Einarbeitung, intuitiv
- Genaue Feinheit zur Algorithmuswahl werden in der Arbeit besorce


	- *Neural Network*: Haben wir vorerst zurückgestellt. Das NN eignet sich zwar gut für Daten mit vielen Dimensionen, jedoch werden extrem viele Daten vorausgesetzt, um versteckte Layers zwischen In- und Output daten zu bilden
	- *Kernel SVM*: Von einer Implementation von Support Vector Machines haben wir vorest abgesehen, da wir als wir uns informiet haben sahen, dass Treebased Methods (folen) in der Praxis mehr Verwendung finden. Je nach Endresultaten werden wir aber noch auf SVMs zurückkehren
	- *Random Forsest*: Unsere erste Versuche haben wir dann mit dem RF gestartet...
	- *Gradient Boosting Tree* ...und sinde deshalb noch nicht zu GBT gelangt. Falls nötig, werden wir aber auch diese noch ausprobieren.

Bevor wir unsere ersten Resulate präsentieren geben wir euch einen kurzen RF Überblick




# ML – Decision Tree

RF besteht aus vielen DT

- Hierzu müssen wir etwas weiter oben mit dem Entscheidungsbaum begebinnen:
- Der DT ist funktioniert ziemlich intuitiv. Jeder Kontenpunkt enthält ein Entscheidungspunk der zu einem nächsten Entscheidungskonten führt. Diese Baumstruktur endet schlussendlich in einer Entscheidung, in unserem Fall einer Klassifikation in "interessante und uninteressante Ausschreibung".
- Im Lernprozess werden die Bäume von den Leafnodes aus rekursiv aufgebaut, also Entscheidungsregeln aus den Ergebnissen abgeleitet (induziert).

# ML – Random Forest
Der Random Forest besteht nur aus vielen solcher zufällig gewählter Entscheidungsbäume mit welchen eine Klassifikation erstellt wird. Am Schluss wird per Mehrheitsprinzp die häufigste Klassifikation gewählt

# ML – Success Metrics
- Wie messen wir den Erfolg unseres Waldes?
- Wir können nicht Wahllos Ausschreibungen für einzelne Anbieter testen, da wir in den meisten Fällen **zu wenig über die Präferenzen, Arbeitsweisen, Expertiesen und das Arbeitsumfeld der einzelnen Anbieter wissen.** Deshalb verwenden wir einige herkömmliche **ML Erfolgskriterien**:
- Bevor ein erstes Modell für einen Anbieter trainiert wird, werden ca. **25% der Daten** für anschliessende testzwecken zurückgehalten. Diese werden klassifiziert (in für den Anbieter interessante / uninteressante Ausschreibungen). Da wir die "Lösungen" haben können wir dann unser System evaluieren.
- **Tabelle die die Wirklichkeit und die Einstufung des System aufzeigen**
- *True Positives* sind Ausschreibungen, für die sich der Anbieter interessiert (und vom System als solche erkannt wurden)
- *True Negatives* sind die für die sich der Anbieter (wahrscheinlich) nicht interessiert und richig erkannt wurden
- *False Negatives* hätten vom System als interessant erkennt werden sollen§
- Und *False Positives* wurden als interessant erkannt obwohl sie das nicht waren.
- Uns interssieren vor allem die letzten beiden: Wir wollen FN verringern, aber nicht zwingend die FPs. Diese könnten ja für den Anbieter ebenfalls interessante Ausschreibungen sein, da sie den Positiven Response Pool Ähnlich sind. Das könnten also Ausschreibungen sein, für welchen der Anbieter entweder keinen Zuschlag erhalten hat oder eine interessante Ausschreibung für die er sich nicht beworben hat.
- In einer Anwendung würde man diese FP dem User evt. noch einmal aufzeigen, damit er/sie diese persöndlich als Interessant/Uninteressant markieren kann und das System so weiter verbessern
- Es gibt dann noch weitere Evaluierungskriterien wie die Genauigkeit

### Model generation process -- Jan (5m)

# Current progress -- Alex & Jan ()

current attributes:
 - zip
 - cpv


