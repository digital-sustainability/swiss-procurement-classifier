import configparser
import sqlalchemy

# git update-index --skip-worktree config.ini


config = configparser.ConfigParser()


config.read("config.ini")

connection_string = 'mysql+' + config['database']['connector'] + '://' + config['database']['user'] + ':' + config['database']['password'] + '@' + config['database']['host'] + '/' + config['database']['database']

if __name__ == "__main__":
    for item, element in config['database'].items():
        print('%s: %s' % (item, element))
    print(connection_string)
else:
    engine = sqlalchemy.create_engine(connection_string)
    connection = engine.connect()
