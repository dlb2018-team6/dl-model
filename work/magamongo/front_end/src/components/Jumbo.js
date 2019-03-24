import React from 'react'

import CssBaseline from "@material-ui/core/CssBaseline";
import Typography from "@material-ui/core/Typography";
import { withStyles } from "@material-ui/core/styles";

import bgImg from "../static/jumbobg.jpg";

const styles = theme => ({
    heroUnit: {
        backgroundSize: "cover",
        backgroundImage: "url(" + bgImg + ")",
        backgroundPosition: "center"
    },
    heroContent: {
        maxWidth: 1100,
        color: theme.palette.common.white,
        margin: "0 auto",
        padding: `${theme.spacing.unit * 16}px 0 ${theme.spacing.unit * 22}px`
    },

});

function Jumbo(props) {
    const { classes } = props
    return (
        <div>
            {/* <Grid container spacing={24}> */}
            <CssBaseline />
            <div className={classes.heroUnit}>
                <div className={classes.heroContent}>
                    <Typography
                        variant="h3"
                        align="right"
                        style={{ color: "white" }}
                    >
                        team6デモ
                    </Typography>
                    <Typography
                        variant="subtitle1"
                        align="right"
                        style={{ color: "white" }}
                    >
                        証明写真生成
                    </Typography>
                </div></div>
            {/* </Grid> */}
        </div>
    )
}

export default withStyles(styles)(Jumbo)