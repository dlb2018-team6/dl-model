import React, { useState } from 'react'
import { PhotoPicker } from "aws-amplify-react"
import Grid from "@material-ui/core/Grid";
import Typography from "@material-ui/core/Typography";
import FormGroup from '@material-ui/core/FormGroup';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Switch from '@material-ui/core/Switch';
import Slider from '@material-ui/lab/Slider';
import Button from "@material-ui/core/Button";
import Card from "@material-ui/core/Card";
import CardHeader from '@material-ui/core/CardHeader';
import CardMedia from "@material-ui/core/CardMedia";
import CardActions from "@material-ui/core/CardActions";
import CardContent from "@material-ui/core/CardContent";
import Divider from '@material-ui/core/Divider';
import CircularProgress from '@material-ui/core/CircularProgress';
import IconButton from "@material-ui/core/IconButton"
import GetApp from "@material-ui/icons/GetApp";
import examPic from "../static/syoumeisyashin_woman.png"

export default function Upload() {

    const [smileLevel, setSmileLevel] = useState(0);
    const [hairDarkness, setHairDarkness] = useState(0);
    const [contrast, setcontrast] = useState(0);
    const [brightness, setBrightness] = useState(0);
    const [onSuits, setOnSuits] = useState(0);
    const [pic, setPic] = useState();
    const [returnedPic, setReturnedPic] = useState(examPic);
    const [isLoading, setIsLoading] = useState(false);
    const [isShowing, setIsShowing] = useState(false);
    const [picSelected, setPicSelected] = useState(false);




    const submitPic = () => {
        console.log(hairDarkness)
        //パターン1: 画像をS3に送る
        // const buf = new Buffer((this.state.avatar2).replace(/^data:image\/\w+;base64,/, ""), 'base64')
        // Storage.put('avatar', buf, {
        //     level: 'protected',
        //     contentType: 'image/*',
        // })
        //     .then(result => {
        //         console.log(result)
        //         //Lambdaに返ってきた画像の場所を送る
        //     })
        //     .catch(err => console.log(err));

        //パターン2: 画像をAPIに直接送る
        // const apiName = "team6";
        // const path = "/pic";
        const myInit = {
            body: {
                pic: pic
            }
        };
        // API.put(apiName, path, myInit)
        //     .then(response => {
        //         console.log("sent: ", response);
        //     })
        //     .catch(error => {
        //         console.log(error.response);
        //     });
        //ダミー
        new Promise((resolve, reject) => { // #1
            console.log('#1')
            setIsLoading(true)
            resolve()
        }).then(() => { // #2
            return new Promise((resolve, reject) => {
                setTimeout(() => {
                    console.log('#2')
                    setIsLoading(false)
                    setIsShowing(true)
                    resolve()
                }, 1500)
            })
        }).catch(() => { // エラーハンドリング
            console.error('Something wrong!')
        })
    }
    return (
        <div>

            <Card>
                <CardHeader
                    title="証明写真を生成する"
                    titleTypographyProps={{ align: 'center', variant: "h3" }}
                />
                <CardContent>

                    <Grid container spacing={24} style={{ padding: 12, justify: "space-evenly" }}>
                        {/* アップローダー */}
                        <Grid item xs={12} md={6}>
                            <Card>
                                <PhotoPicker preview headerText="証明写真にする画像を選択してください" headerHint="正面からの顔写真を選択してください" onLoad={dataURL => {
                                    setPic(dataURL)
                                    setPicSelected(true)
                                }} />
                            </Card>
                        </Grid>
                        {/* 設定スライダー */}
                        <Grid item xs={12} md={6}>
                            <Card>
                                <CardHeader
                                    title="生成する証明写真の設定"
                                    subheader="設定を完了してから「生成する」ボタンを押してください"
                                />
                                <CardContent>
                                    <Grid item xs={12}>
                                        <Typography id="label">生成する画像の笑顔レベル</Typography>
                                        <Slider
                                            max={20}
                                            value={smileLevel}
                                            aria-labelledby="label"
                                            style={{ padding: '22px 0px', }}
                                            onChange={(e, v) => setSmileLevel((v - 10) / 10)}
                                        />
                                    </Grid>
                                    <Grid item xs={12}>
                                        <Typography id="label">生成する画像の髪の黒さ</Typography>
                                        <Slider
                                            max={20}
                                            value={hairDarkness}
                                            aria-labelledby="label"
                                            style={{ padding: '22px 0px', }}
                                            onChange={(e, v) => setHairDarkness((v - 10) / 10)}
                                        />
                                    </Grid>
                                    <Grid item xs={12}>
                                        <Typography id="label">画像のコントラスト</Typography>
                                        <Slider
                                            max={20}
                                            value={contrast}
                                            aria-labelledby="label"
                                            style={{ padding: '22px 0px', }}
                                            onChange={(e, v) => setcontrast((v - 10) / 10)}
                                        />
                                    </Grid>
                                    <Grid item xs={12}>
                                        <Typography id="label">画像の明るさ</Typography>
                                        <Slider
                                            max={20}
                                            value={brightness}
                                            aria-labelledby="label"
                                            style={{ padding: '22px 0px', }}
                                            onChange={(e, v) => setBrightness((v - 10) / 10)}
                                        />
                                    </Grid>
                                    <FormGroup>
                                        <FormControlLabel
                                            control={
                                                <Switch
                                                    value={onSuits}
                                                />
                                            }
                                            label="生成する画像にスーツを着せますか？"
                                        />
                                    </FormGroup>
                                </CardContent>
                                <Divider variant="middle" />
                                <CardActions style={{ justifyContent: "center" }}>
                                    <Button size="large"
                                        variant="contained"
                                        color="primary"
                                        onClick={submitPic}
                                        style={{ margin: 22 }}
                                        disabled={!picSelected}
                                    >この内容で生成する</Button>
                                </CardActions>
                            </Card>
                        </Grid>

                    </Grid>
                </CardContent>
            </Card>
            {/* 結果表示エリア */}

            <Grid container spacing={24} style={{ padding: 12 }} justify="space-evenly" >
                {/* タイトル */}
                <Grid item xs={12}><Typography variant="h4" gutterBottom align="center">{isShowing ? ("生成された証明写真") : ("生成された証明写真は以下に表示されます")}</Typography>
                </Grid>
                {/* ローディングのぐるぐる */}
                <Grid container justify="center" alignItems="center"  >
                    {isLoading ? (<CircularProgress disableShrink style={{ margin: 6 }} />) : ("")}
                </Grid>
                {/* 写真カード */}
                {isShowing ? (<Grid container xs={12} md={7} justify="center" alignItems="center">
                    <Card style={{ margin: 5 }}>
                        <CardMedia
                            component="img"
                            // height="140"
                            image={returnedPic}
                            title="生成された証明写真"
                        />
                        <CardActions>
                            <IconButton size="large">
                                <GetApp />
                            </IconButton>
                        </CardActions>
                    </Card>
                </Grid>) : ("")}

            </Grid>



        </div >
    )
}
