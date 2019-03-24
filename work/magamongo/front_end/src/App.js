import React, { Component } from 'react';
import { I18n } from "aws-amplify";
import Jumbo from "./components/Jumbo"
import Uploader from "./components/Upload"
import Typography from "@material-ui/core/Typography";
import { withStyles } from "@material-ui/core/styles";


const dict = {
  ja: {
    "Select a Photo": "画像を選択",
  }
};

I18n.putVocabularies(dict);

const styles = theme => ({
  footer: {
    backgroundColor: theme.palette.background.paper,
    padding: theme.spacing.unit * 6
  }
});
function App(props) {
  const { classes } = props;

  return (
    <div className="App">
      <Jumbo />
      <Uploader />
      <footer className={classes.footer}>
        <Typography
          variant="subtitle1"
          align="center"
          color="textSecondary"
          component="p"
        >
          Copyright &copy; {new Date().getFullYear()} DL基礎講座team6
      </Typography>
      </footer>
    </div>
  );
}


export default withStyles(styles)(App);