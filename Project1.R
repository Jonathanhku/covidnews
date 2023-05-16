library(readr)

# import dataset
# May 2020
c1_F = read_csv(url("https://raw.githubusercontent.com/cuilimeng/CoAID/master/05-01-2020/NewsFakeCOVID-19.csv"))
c1_R = read_csv(url("https://raw.githubusercontent.com/cuilimeng/CoAID/master/05-01-2020/NewsRealCOVID-19.csv"))
#Jul 2020
c2_F = read_csv(url("https://raw.githubusercontent.com/cuilimeng/CoAID/master/07-01-2020/NewsFakeCOVID-19.csv"))
c2_R = read_csv(url("https://raw.githubusercontent.com/cuilimeng/CoAID/master/07-01-2020/NewsRealCOVID-19.csv"))
#Sep 2020
c3_F = read_csv(url("https://raw.githubusercontent.com/cuilimeng/CoAID/master/09-01-2020/NewsFakeCOVID-19.csv"))
c3_R = read_csv(url("https://raw.githubusercontent.com/cuilimeng/CoAID/master/09-01-2020/NewsRealCOVID-19.csv"))
#Nov2020
c4_F = read_csv(url("https://raw.githubusercontent.com/cuilimeng/CoAID/master/11-01-2020/NewsFakeCOVID-19.csv"))
c4_R = read_csv(url("https://raw.githubusercontent.com/cuilimeng/CoAID/master/11-01-2020/NewsRealCOVID-19.csv"))

Fake = dim(c1_F)[1]+dim(c2_F)[1]+dim(c3_F)[1]+dim(c4_F)[1]
Real = dim(c1_R)[1]+dim(c2_R)[1]+dim(c3_R)[1]+dim(c4_R)[1]

Fake_df = rbind(c1_F,c2_F)
Fake_df = rbind(Fake_df, c3_F)
Fake_df = rbind(Fake_df, c4_F)
Real_df = rbind(c1_R,c2_R)
Real_df = rbind(Real_df, c3_R)
Real_df = rbind(Real_df, c4_R)

#Find the portion of missing data
colMeans(is.na(Fake_df))
colMeans(is.na(Real_df))

write.csv(Real_df['title'], "real1.csv")
write.csv(Fake_df['title'], "fake1.csv")

rd = read_csv(url("https://raw.githubusercontent.com/apurvamulay/ReCOVery/master/dataset/recovery-news-data.csv"))

colMeans(is.na(rd[c("publish_date","title", "body_text" )]))

rdc = rd[c("publisher","publish_date","title", "body_text", "reliability")]
rdc = na.omit(rdc)
real2 = rdc[rdc$reliability==1,]
fake2 = rdc[rdc$reliability==0,]

write.csv(real2, "real2.csv")
write.csv(fake2, "fake2.csv")

fd = read_csv(url("https://raw.githubusercontent.com/merry555/FibVID/main/claim_propagation/claim_propagation.csv"))

fdc = fd[1:ncol(fd)-1]
real3 = fdc[fd$group==0,]
fake3 = fdc[fd$group==1,]

# removing missing data
real3 =na.omit(real3)
fake3 =na.omit(fake3)

write.csv(real3, "real3.csv")
write.csv(fake3, "fake3.csv")

