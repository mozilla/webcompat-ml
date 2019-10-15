import re


class IssueParser:
    def __init__(self, df):
        self.df = df

    def get_url(self):
        """Extract URL from body"""
        expr = "\*\*URL\*\*: ([^\n]*)"
        url = self.df.body.str.extract(expr)
        return url

    def get_problem_type(self):
        """Extract problem type from body"""
        expr = "\*\*Problem type\*\*: ([^\n]*)"
        problem_type = self.df.body.str.extract(expr)
        return problem_type

    def get_browser(self):
        """Extract browser information from body"""
        expr = "\*\*Browser / Version\*\*: ([^\n]*)"
        browser = self.df.body.str.extract(expr)
        return browser

    def get_os(self):
        """Extract OS information from body"""
        expr = "\*\*Operating System\*\*: ([^\n]*)"
        os = self.df.body.str.extract(expr)
        return os

    def get_other_browser_test(self):
        """Extract if issue has been tested in other browsers"""
        expr = "\*\*Tested Another Browser\*\*: ([^\n]*)"
        other_browser_test = self.df.body.str.extract(expr)
        return os

    def get_reproduction_steps(self):
        """Extract reproduction steps from body"""
        body = self.df.body

        # This field has no specific format
        # Cleaning up the rest of the fields first
        body = body.str.replace("<!--.*\n", "")
        body = body.str.replace("\[\!\[Screenshot Description\].*\n", "")
        body = body.str.replace("\_From \[webcompat\.com.*", "")
        body = body.str.replace("<details>.*</details>", "", flags=re.S)
        body = body.str.replace("Submitted in the name of.*", "")
        body = body.str.replace("\n", "")

        # Extracting the remaining values
        steps_to_reproduce = body.str.extract("\*\*Steps to Reproduce\*\*:(.*)", flags=re.S)
        return steps_to_reproduce
