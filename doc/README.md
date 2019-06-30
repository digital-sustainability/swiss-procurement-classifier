# Configure Database connection

Copy config.ini.default to config.ini
pandoc -t revealjs -V revealjs-url=http://lab.hakim.se/reveal-js -V theme=white -s doc/presentation.md -o doc/presentation.html --css test.css
