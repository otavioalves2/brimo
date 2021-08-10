import twint

# Configure
c = twint.Config()
c.Search = "alegria"
c.Lowercase = True
c.Links = 'exclude'
c.Lang = "pt"
c.Filter_retweets = True
c.Limit = 1000
c.Store_json = True
c.Output = "alegria.json"

# Run
twint.run.Search(c)

# Configure
c = twint.Config()
c.Search = "tristeza"
c.Lowercase = True
c.Links = 'exclude'
c.Lang = "pt"
c.Filter_retweets = True
c.Limit = 1000
c.Store_json = True
c.Output = "tristeza.json"

# Run
twint.run.Search(c)

# Configure
c = twint.Config()
c.Search = "raiva"
c.Lowercase = True
c.Links = 'exclude'
c.Lang = "pt"
c.Filter_retweets = True
c.Limit = 1000
c.Store_json = True
c.Output = "raiva.json"

# Run
twint.run.Search(c)

# Configure
c = twint.Config()
c.Search = "nojo"
c.Lowercase = True
c.Links = 'exclude'
c.Lang = "pt"
c.Filter_retweets = True
c.Limit = 1000
c.Store_json = True
c.Output = "nojo.json"

# Run
twint.run.Search(c)

# Configure
c = twint.Config()
c.Search = "surpresa"
c.Lowercase = True
c.Links = 'exclude'
c.Lang = "pt"
c.Filter_retweets = True
c.Limit = 1000
c.Store_json = True
c.Output = "surpresa.json"

# Run
twint.run.Search(c)

# Configure
c = twint.Config()
c.Search = "medo"
c.Lowercase = True
c.Links = 'exclude'
c.Lang = "pt"
c.Filter_retweets = True
c.Limit = 1000
c.Store_json = True
c.Output = "medo.json"

# Run
twint.run.Search(c)