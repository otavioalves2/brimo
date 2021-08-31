import twint

# Configure
c = twint.Config()
c.Search = "bolsonaro"
c.Lowercase = True
c.Links = 'exclude'
c.Lang = "pt"
c.Filter_retweets = True
c.Limit = 10
c.Store_json = True
c.Output = "teste.json"

# Run
twint.run.Search(c)