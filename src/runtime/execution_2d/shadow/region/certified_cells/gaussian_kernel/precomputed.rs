//! Derived canonical finite-kernel coefficients; verified bit-for-bit against the interval proof.
use super::{Interval, Patch, Polynomial};
pub(super) const NORMALIZATION: Interval = Interval {
    lo: f64::from_bits(4612811605085194079),
    hi: f64::from_bits(4612811605085194881),
};
#[cfg(test)]
pub(super) const REMAINDER: f64 = f64::from_bits(4421083568775373667);
pub(super) const DATA: [Patch; 16] = [
    Patch {
        center: f64::from_bits(4591419820104220672),
        cdf: Polynomial {
            coefficients: [
                f64::from_bits(4591409937531895293),
                f64::from_bits(4607142923227243077),
                f64::from_bits(13810258635652045236),
                f64::from_bits(13818465981127860816),
                f64::from_bits(4577859695115492864),
                f64::from_bits(4582705460749973067),
                f64::from_bits(13789930064479173761),
                f64::from_bits(13792171368246158036),
                f64::from_bits(4553021048479421906),
                f64::from_bits(4553763811314623680),
                f64::from_bits(13761083574202345440),
                f64::from_bits(13760646097051688642),
                f64::from_bits(4521823387360873790),
            ],
            error: f64::from_bits(4421108567470893246),
        },
        #[cfg(test)]
        density: Polynomial {
            coefficients: [
                f64::from_bits(4607142923227243077),
                f64::from_bits(13814762235279415732),
                f64::from_bits(13825932542746506104),
                f64::from_bits(4586866894370233856),
                f64::from_bits(4593474719840101342),
                f64::from_bits(13801212062179921249),
                f64::from_bits(13804850553520135801),
                f64::from_bits(4566531847361533394),
                f64::from_bits(4567915633655538008),
                f64::from_bits(13776366951137669592),
                f64::from_bits(13776581484604235045),
                f64::from_bits(4537690666797118428),
                f64::from_bits(0),
            ],
            error: f64::from_bits(4427601924488991634),
        },
        #[cfg(test)]
        oracle_coefficients: [
            Interval {
                lo: f64::from_bits(4591409937531895261),
                hi: f64::from_bits(4591409937531895325),
            },
            Interval {
                lo: f64::from_bits(4607142923227243036),
                hi: f64::from_bits(4607142923227243118),
            },
            Interval {
                lo: f64::from_bits(13810258635652045268),
                hi: f64::from_bits(13810258635652045203),
            },
            Interval {
                lo: f64::from_bits(13818465981127860846),
                hi: f64::from_bits(13818465981127860786),
            },
            Interval {
                lo: f64::from_bits(4577859695115492829),
                hi: f64::from_bits(4577859695115492899),
            },
            Interval {
                lo: f64::from_bits(4582705460749973030),
                hi: f64::from_bits(4582705460749973104),
            },
            Interval {
                lo: f64::from_bits(13789930064479173810),
                hi: f64::from_bits(13789930064479173712),
            },
            Interval {
                lo: f64::from_bits(13792171368246158073),
                hi: f64::from_bits(13792171368246157998),
            },
            Interval {
                lo: f64::from_bits(4553021048479421855),
                hi: f64::from_bits(4553021048479421957),
            },
            Interval {
                lo: f64::from_bits(4553763811314623649),
                hi: f64::from_bits(4553763811314623712),
            },
            Interval {
                lo: f64::from_bits(13761083574202345483),
                hi: f64::from_bits(13761083574202345397),
            },
            Interval {
                lo: f64::from_bits(13760646097051688686),
                hi: f64::from_bits(13760646097051688597),
            },
            Interval {
                lo: f64::from_bits(4521823387360873758),
                hi: f64::from_bits(4521823387360873821),
            },
        ],
    },
    Patch {
        center: f64::from_bits(4598738169498697728),
        cdf: Polynomial {
            coefficients: [
                f64::from_bits(4598672159303344908),
                f64::from_bits(4606833129887792440),
                f64::from_bits(13817410131712976496),
                f64::from_bits(13817855427882936630),
                f64::from_bits(4584902485434195084),
                f64::from_bits(4581502229048754340),
                f64::from_bits(13796175773299314732),
                f64::from_bits(13790590638425887714),
                f64::from_bits(4559127110699051505),
                f64::from_bits(4551273594380305725),
                f64::from_bits(13767660227858944850),
                f64::from_bits(13757885493121301410),
                f64::from_bits(4527655782404037530),
            ],
            error: f64::from_bits(4421132135887214256),
        },
        #[cfg(test)]
        density: Polynomial {
            coefficients: [
                f64::from_bits(4606833129887792440),
                f64::from_bits(13821913731340346992),
                f64::from_bits(13825016712879119825),
                f64::from_bits(4593909684688936076),
                f64::from_bits(4591970680213577933),
                f64::from_bits(13808148143608712224),
                f64::from_bits(13803401971964228298),
                f64::from_bits(4572637909581162993),
                f64::from_bits(4565677089557851620),
                f64::from_bits(13782335968394733606),
                f64::from_bits(13773276194518540543),
                f64::from_bits(4544159741782504628),
                f64::from_bits(0),
            ],
            error: f64::from_bits(4427602290869736394),
        },
        #[cfg(test)]
        oracle_coefficients: [
            Interval {
                lo: f64::from_bits(4598672159303344884),
                hi: f64::from_bits(4598672159303344933),
            },
            Interval {
                lo: f64::from_bits(4606833129887792399),
                hi: f64::from_bits(4606833129887792481),
            },
            Interval {
                lo: f64::from_bits(13817410131712976521),
                hi: f64::from_bits(13817410131712976470),
            },
            Interval {
                lo: f64::from_bits(13817855427882936662),
                hi: f64::from_bits(13817855427882936598),
            },
            Interval {
                lo: f64::from_bits(4584902485434195056),
                hi: f64::from_bits(4584902485434195112),
            },
            Interval {
                lo: f64::from_bits(4581502229048754297),
                hi: f64::from_bits(4581502229048754383),
            },
            Interval {
                lo: f64::from_bits(13796175773299314772),
                hi: f64::from_bits(13796175773299314691),
            },
            Interval {
                lo: f64::from_bits(13790590638425887759),
                hi: f64::from_bits(13790590638425887668),
            },
            Interval {
                lo: f64::from_bits(4559127110699051462),
                hi: f64::from_bits(4559127110699051548),
            },
            Interval {
                lo: f64::from_bits(4551273594380305647),
                hi: f64::from_bits(4551273594380305803),
            },
            Interval {
                lo: f64::from_bits(13767660227858944887),
                hi: f64::from_bits(13767660227858944813),
            },
            Interval {
                lo: f64::from_bits(13757885493121301521),
                hi: f64::from_bits(13757885493121301298),
            },
            Interval {
                lo: f64::from_bits(4527655782404037479),
                hi: f64::from_bits(4527655782404037582),
            },
        ],
    },
    Patch {
        center: f64::from_bits(4602115869219225600),
        cdf: Polynomial {
            coefficients: [
                f64::from_bits(4601816562721408644),
                f64::from_bits(4606245279354160934),
                f64::from_bits(13820105738216140468),
                f64::from_bits(13816432261928485958),
                f64::from_bits(4587172375075068792),
                f64::from_bits(4578599254308979821),
                f64::from_bits(13798664430066173234),
                f64::from_bits(13785791017532712220),
                f64::from_bits(4561106675882195496),
                f64::from_bits(4543344438454552200),
                f64::from_bits(13768858852518818856),
                f64::from_bits(13741521875853234091),
                f64::from_bits(4528769237193013199),
            ],
            error: f64::from_bits(4421153164131307129),
        },
        #[cfg(test)]
        density: Polynomial {
            coefficients: [
                f64::from_bits(4606245279354160934),
                f64::from_bits(13824609337843510964),
                f64::from_bits(13823340510080590644),
                f64::from_bits(4596179574329809784),
                f64::from_bits(4589317940993118916),
                f64::from_bits(13810014636183856101),
                f64::from_bits(13798380335121699056),
                f64::from_bits(4574617474764306984),
                f64::from_bits(4557762755973581740),
                f64::from_bits(13783834249219576114),
                f64::from_bits(13757111304375303564),
                f64::from_bits(4544994832874236380),
                f64::from_bits(0),
            ],
            error: f64::from_bits(4427602348524539914),
        },
        #[cfg(test)]
        oracle_coefficients: [
            Interval {
                lo: f64::from_bits(4601816562721408604),
                hi: f64::from_bits(4601816562721408683),
            },
            Interval {
                lo: f64::from_bits(4606245279354160893),
                hi: f64::from_bits(4606245279354160975),
            },
            Interval {
                lo: f64::from_bits(13820105738216140508),
                hi: f64::from_bits(13820105738216140427),
            },
            Interval {
                lo: f64::from_bits(13816432261928486028),
                hi: f64::from_bits(13816432261928485887),
            },
            Interval {
                lo: f64::from_bits(4587172375075068746),
                hi: f64::from_bits(4587172375075068838),
            },
            Interval {
                lo: f64::from_bits(4578599254308979716),
                hi: f64::from_bits(4578599254308979926),
            },
            Interval {
                lo: f64::from_bits(13798664430066173301),
                hi: f64::from_bits(13798664430066173166),
            },
            Interval {
                lo: f64::from_bits(13785791017532712339),
                hi: f64::from_bits(13785791017532712100),
            },
            Interval {
                lo: f64::from_bits(4561106675882195422),
                hi: f64::from_bits(4561106675882195570),
            },
            Interval {
                lo: f64::from_bits(4543344438454551764),
                hi: f64::from_bits(4543344438454552637),
            },
            Interval {
                lo: f64::from_bits(13768858852518818920),
                hi: f64::from_bits(13768858852518818791),
            },
            Interval {
                lo: f64::from_bits(13741521875853235410),
                hi: f64::from_bits(13741521875853232772),
            },
            Interval {
                lo: f64::from_bits(4528769237193013106),
                hi: f64::from_bits(4528769237193013292),
            },
        ],
    },
    Patch {
        center: f64::from_bits(4604086194056200192),
        cdf: Polynomial {
            coefficients: [
                f64::from_bits(4603687979576299498),
                f64::from_bits(4605437487303959852),
                f64::from_bits(13821809519989317733),
                f64::from_bits(13813549350881628813),
                f64::from_bits(4588324223459602590),
                f64::from_bits(4571309393681535763),
                f64::from_bits(13799134282642289716),
                f64::from_bits(4553586502277196042),
                f64::from_bits(4560836658186929390),
                f64::from_bits(13769415973302137104),
                f64::from_bits(13768053174760801644),
                f64::from_bits(4532728392123608514),
                f64::from_bits(4527015474513525594),
            ],
            error: f64::from_bits(4421174850720834894),
        },
        #[cfg(test)]
        density: Polynomial {
            coefficients: [
                f64::from_bits(4605437487303959852),
                f64::from_bits(13826313119616688229),
                f64::from_bits(13820809397190843347),
                f64::from_bits(4597331422714343582),
                f64::from_bits(4581481435818239960),
                f64::from_bits(13810445793900260942),
                f64::from_bits(4566306860506683346),
                f64::from_bits(4574347457069040878),
                f64::from_bits(13783728711191486258),
                f64::from_bits(13782827152022054599),
                f64::from_bits(4548676926065509237),
                f64::from_bits(4543226997712410888),
                f64::from_bits(0),
            ],
            error: f64::from_bits(4427602303343212489),
        },
        #[cfg(test)]
        oracle_coefficients: [
            Interval {
                lo: f64::from_bits(4603687979576299470),
                hi: f64::from_bits(4603687979576299526),
            },
            Interval {
                lo: f64::from_bits(4605437487303959811),
                hi: f64::from_bits(4605437487303959893),
            },
            Interval {
                lo: f64::from_bits(13821809519989317762),
                hi: f64::from_bits(13821809519989317704),
            },
            Interval {
                lo: f64::from_bits(13813549350881628895),
                hi: f64::from_bits(13813549350881628731),
            },
            Interval {
                lo: f64::from_bits(4588324223459602522),
                hi: f64::from_bits(4588324223459602657),
            },
            Interval {
                lo: f64::from_bits(4571309393681535495),
                hi: f64::from_bits(4571309393681536031),
            },
            Interval {
                lo: f64::from_bits(13799134282642289768),
                hi: f64::from_bits(13799134282642289664),
            },
            Interval {
                lo: f64::from_bits(4553586502277195374),
                hi: f64::from_bits(4553586502277196711),
            },
            Interval {
                lo: f64::from_bits(4560836658186929272),
                hi: f64::from_bits(4560836658186929509),
            },
            Interval {
                lo: f64::from_bits(13769415973302137436),
                hi: f64::from_bits(13769415973302136771),
            },
            Interval {
                lo: f64::from_bits(13768053174760801752),
                hi: f64::from_bits(13768053174760801536),
            },
            Interval {
                lo: f64::from_bits(4532728392123608242),
                hi: f64::from_bits(4532728392123608785),
            },
            Interval {
                lo: f64::from_bits(4527015474513525432),
                hi: f64::from_bits(4527015474513525757),
            },
        ],
    },
    Patch {
        center: f64::from_bits(4605775043916464128),
        cdf: Polynomial {
            coefficients: [
                f64::from_bits(4604961973298667426),
                f64::from_bits(4604484787116632796),
                f64::from_bits(13822367354411013674),
                f64::from_bits(13808380051727940376),
                f64::from_bits(4588281539503640622),
                f64::from_bits(13795168396335206946),
                f64::from_bits(13797960455020920723),
                f64::from_bits(4563920900655782875),
                f64::from_bits(4558619280683990941),
                f64::from_bits(13774442031767695408),
                f64::from_bits(13763949155870431798),
                f64::from_bits(4535998873953331274),
                f64::from_bits(4519774654743247289),
            ],
            error: f64::from_bits(4421196433460689957),
        },
        #[cfg(test)]
        density: Polynomial {
            coefficients: [
                f64::from_bits(4604484787116632796),
                f64::from_bits(13826870954038384170),
                f64::from_bits(13815307248273995940),
                f64::from_bits(4597288738758381614),
                f64::from_bits(13805462179921634986),
                f64::from_bits(13809486654899916718),
                f64::from_bits(4576646939818289248),
                f64::from_bits(4572130079566102429),
                f64::from_bits(13788820077011818038),
                f64::from_bits(13778823028315934915),
                f64::from_bits(4551526291142422374),
                f64::from_bits(4535997096223226699),
                f64::from_bits(0),
            ],
            error: f64::from_bits(4427603288492112346),
        },
        #[cfg(test)]
        oracle_coefficients: [
            Interval {
                lo: f64::from_bits(4604961973298667390),
                hi: f64::from_bits(4604961973298667462),
            },
            Interval {
                lo: f64::from_bits(4604484787116632753),
                hi: f64::from_bits(4604484787116632839),
            },
            Interval {
                lo: f64::from_bits(13822367354411013712),
                hi: f64::from_bits(13822367354411013635),
            },
            Interval {
                lo: f64::from_bits(13808380051727940578),
                hi: f64::from_bits(13808380051727940174),
            },
            Interval {
                lo: f64::from_bits(4588281539503640526),
                hi: f64::from_bits(4588281539503640718),
            },
            Interval {
                lo: f64::from_bits(13795168396335207319),
                hi: f64::from_bits(13795168396335206572),
            },
            Interval {
                lo: f64::from_bits(13797960455020920880),
                hi: f64::from_bits(13797960455020920566),
            },
            Interval {
                lo: f64::from_bits(4563920900655782618),
                hi: f64::from_bits(4563920900655783132),
            },
            Interval {
                lo: f64::from_bits(4558619280683990750),
                hi: f64::from_bits(4558619280683991132),
            },
            Interval {
                lo: f64::from_bits(13774442031767695682),
                hi: f64::from_bits(13774442031767695135),
            },
            Interval {
                lo: f64::from_bits(13763949155870432163),
                hi: f64::from_bits(13763949155870431432),
            },
            Interval {
                lo: f64::from_bits(4535998873953331037),
                hi: f64::from_bits(4535998873953331512),
            },
            Interval {
                lo: f64::from_bits(4519774654743246136),
                hi: f64::from_bits(4519774654743248442),
            },
        ],
    },
    Patch {
        center: f64::from_bits(4607323156288372736),
        cdf: Polynomial {
            coefficients: [
                f64::from_bits(4606050055014946771),
                f64::from_bits(4603467693979861516),
                f64::from_bits(13822501521033347612),
                f64::from_bits(4573816914617183594),
                f64::from_bits(4587207009075443162),
                f64::from_bits(13800876280703301999),
                f64::from_bits(13795354966581723640),
                f64::from_bits(4566741831396993591),
                f64::from_bits(4552938247757958792),
                f64::from_bits(13775661521140781226),
                f64::from_bits(4520090045390397417),
                f64::from_bits(4535916806689590470),
                f64::from_bits(13743827198630473258),
            ],
            error: f64::from_bits(4421219432939974401),
        },
        #[cfg(test)]
        density: Polynomial {
            coefficients: [
                f64::from_bits(4603467693979861516),
                f64::from_bits(13827005120660718108),
                f64::from_bits(4581032390756049424),
                f64::from_bits(4596214208330184154),
                f64::from_bits(13811471135474911178),
                f64::from_bits(13807028619623097075),
                f64::from_bits(4579195587304745440),
                f64::from_bits(4566449046640070280),
                f64::from_bits(13790107030782499168),
                f64::from_bits(4535298236862944242),
                f64::from_bits(4551413448654778769),
                f64::from_bits(13759879513352340128),
                f64::from_bits(0),
            ],
            error: f64::from_bits(4427603607124548943),
        },
        #[cfg(test)]
        oracle_coefficients: [
            Interval {
                lo: f64::from_bits(4606050055014946726),
                hi: f64::from_bits(4606050055014946816),
            },
            Interval {
                lo: f64::from_bits(4603467693979861472),
                hi: f64::from_bits(4603467693979861560),
            },
            Interval {
                lo: f64::from_bits(13822501521033347660),
                hi: f64::from_bits(13822501521033347565),
            },
            Interval {
                lo: f64::from_bits(4573816914617181607),
                hi: f64::from_bits(4573816914617185582),
            },
            Interval {
                lo: f64::from_bits(4587207009075443033),
                hi: f64::from_bits(4587207009075443291),
            },
            Interval {
                lo: f64::from_bits(13800876280703302257),
                hi: f64::from_bits(13800876280703301741),
            },
            Interval {
                lo: f64::from_bits(13795354966581723867),
                hi: f64::from_bits(13795354966581723412),
            },
            Interval {
                lo: f64::from_bits(4566741831396993400),
                hi: f64::from_bits(4566741831396993782),
            },
            Interval {
                lo: f64::from_bits(4552938247757957619),
                hi: f64::from_bits(4552938247757959965),
            },
            Interval {
                lo: f64::from_bits(13775661521140781661),
                hi: f64::from_bits(13775661521140780792),
            },
            Interval {
                lo: f64::from_bits(4520090045390378321),
                hi: f64::from_bits(4520090045390416513),
            },
            Interval {
                lo: f64::from_bits(4535916806689590072),
                hi: f64::from_bits(4535916806689590869),
            },
            Interval {
                lo: f64::from_bits(13743827198630475257),
                hi: f64::from_bits(13743827198630471260),
            },
        ],
    },
    Patch {
        center: f64::from_bits(4608167581218504704),
        cdf: Polynomial {
            coefficients: [
                f64::from_bits(4606947343679196049),
                f64::from_bits(4602243572741733983),
                f64::from_bits(13822267190524701954),
                f64::from_bits(4585708035298089416),
                f64::from_bits(4585435365859611730),
                f64::from_bits(13802996668396428104),
                f64::from_bits(13789754193474365091),
                f64::from_bits(4566941026910314684),
                f64::from_bits(13771392442666565540),
                f64::from_bits(13774394127282361729),
                f64::from_bits(4539975639945746937),
                f64::from_bits(4532987555715943792),
                f64::from_bits(13749153148234816008),
            ],
            error: f64::from_bits(4421235737692456974),
        },
        #[cfg(test)]
        density: Polynomial {
            coefficients: [
                f64::from_bits(4602243572741733983),
                f64::from_bits(13826770790152072450),
                f64::from_bits(4592985242056607404),
                f64::from_bits(4594442565114352722),
                f64::from_bits(13813330838618315021),
                f64::from_bits(13801080158926314746),
                f64::from_bits(4579544179453057354),
                f64::from_bits(13784903241548677028),
                f64::from_bits(13788766184465817650),
                f64::from_bits(4554699142623772792),
                f64::from_bits(4548855101035239741),
                f64::from_bits(13764999875462439814),
                f64::from_bits(0),
            ],
            error: f64::from_bits(4427592786249714358),
        },
        #[cfg(test)]
        oracle_coefficients: [
            Interval {
                lo: f64::from_bits(4606947343679195996),
                hi: f64::from_bits(4606947343679196102),
            },
            Interval {
                lo: f64::from_bits(4602243572741733929),
                hi: f64::from_bits(4602243572741734037),
            },
            Interval {
                lo: f64::from_bits(13822267190524701989),
                hi: f64::from_bits(13822267190524701919),
            },
            Interval {
                lo: f64::from_bits(4585708035298089228),
                hi: f64::from_bits(4585708035298089604),
            },
            Interval {
                lo: f64::from_bits(4585435365859611625),
                hi: f64::from_bits(4585435365859611836),
            },
            Interval {
                lo: f64::from_bits(13802996668396428324),
                hi: f64::from_bits(13802996668396427885),
            },
            Interval {
                lo: f64::from_bits(13789754193474365898),
                hi: f64::from_bits(13789754193474364284),
            },
            Interval {
                lo: f64::from_bits(4566941026910314508),
                hi: f64::from_bits(4566941026910314861),
            },
            Interval {
                lo: f64::from_bits(13771392442666567784),
                hi: f64::from_bits(13771392442666563295),
            },
            Interval {
                lo: f64::from_bits(13774394127282362157),
                hi: f64::from_bits(13774394127282361301),
            },
            Interval {
                lo: f64::from_bits(4539975639945746327),
                hi: f64::from_bits(4539975639945747547),
            },
            Interval {
                lo: f64::from_bits(4532987555715942959),
                hi: f64::from_bits(4532987555715944625),
            },
            Interval {
                lo: f64::from_bits(13749153148234817089),
                hi: f64::from_bits(13749153148234814928),
            },
        ],
    },
    Patch {
        center: f64::from_bits(4609012006148636672),
        cdf: Polynomial {
            coefficients: [
                f64::from_bits(4607422112345300827),
                f64::from_bits(4600373592002533906),
                f64::from_bits(13821755980894685781),
                f64::from_bits(4588896073713805650),
                f64::from_bits(4582081444527874774),
                f64::from_bits(13803380435747327074),
                f64::from_bits(4558934618796698065),
                f64::from_bits(4565493817174853504),
                f64::from_bits(13778769761318897066),
                f64::from_bits(13770986805994314670),
                f64::from_bits(4542179589250548040),
                f64::from_bits(4523589928780966977),
                f64::from_bits(13749641335927514897),
            ],
            error: f64::from_bits(4421262139868719899),
        },
        #[cfg(test)]
        density: Polynomial {
            coefficients: [
                f64::from_bits(4600373592002533906),
                f64::from_bits(13826259580522056277),
                f64::from_bits(4595719459799043837),
                f64::from_bits(4591088643782615766),
                f64::from_bits(13813570693212626878),
                f64::from_bits(4570839515814002362),
                f64::from_bits(4578023241772476048),
                f64::from_bits(13792280560201008554),
                f64::from_bits(13785495897970186020),
                f64::from_bits(4557454079254774170),
                f64::from_bits(4539530541111213273),
                f64::from_bits(13765479771406006938),
                f64::from_bits(0),
            ],
            error: f64::from_bits(4427593303472500230),
        },
        #[cfg(test)]
        oracle_coefficients: [
            Interval {
                lo: f64::from_bits(4607422112345300795),
                hi: f64::from_bits(4607422112345300859),
            },
            Interval {
                lo: f64::from_bits(4600373592002533849),
                hi: f64::from_bits(4600373592002533963),
            },
            Interval {
                lo: f64::from_bits(13821755980894685823),
                hi: f64::from_bits(13821755980894685739),
            },
            Interval {
                lo: f64::from_bits(4588896073713805414),
                hi: f64::from_bits(4588896073713805885),
            },
            Interval {
                lo: f64::from_bits(4582081444527874494),
                hi: f64::from_bits(4582081444527875053),
            },
            Interval {
                lo: f64::from_bits(13803380435747327377),
                hi: f64::from_bits(13803380435747326772),
            },
            Interval {
                lo: f64::from_bits(4558934618796695737),
                hi: f64::from_bits(4558934618796700393),
            },
            Interval {
                lo: f64::from_bits(4565493817174852980),
                hi: f64::from_bits(4565493817174854028),
            },
            Interval {
                lo: f64::from_bits(13778769761318897937),
                hi: f64::from_bits(13778769761318896195),
            },
            Interval {
                lo: f64::from_bits(13770986805994316032),
                hi: f64::from_bits(13770986805994313309),
            },
            Interval {
                lo: f64::from_bits(4542179589250547035),
                hi: f64::from_bits(4542179589250549045),
            },
            Interval {
                lo: f64::from_bits(4523589928780961356),
                hi: f64::from_bits(4523589928780972598),
            },
            Interval {
                lo: f64::from_bits(13749641335927515838),
                hi: f64::from_bits(13749641335927513956),
            },
        ],
    },
    Patch {
        center: f64::from_bits(4609856431078768640),
        cdf: Polynomial {
            coefficients: [
                f64::from_bits(4607696757997407254),
                f64::from_bits(4598730533858807947),
                f64::from_bits(13820602701238623838),
                f64::from_bits(4589858370718374886),
                f64::from_bits(4576098290228905312),
                f64::from_bits(13802334511769833982),
                f64::from_bits(4567806172475671480),
                f64::from_bits(4562502618317482378),
                f64::from_bits(13780731524215433910),
                f64::from_bits(13759514375174895532),
                f64::from_bits(4541768424889642840),
                f64::from_bits(13750879689476468588),
                f64::from_bits(13746969817143573931),
            ],
            error: f64::from_bits(4421292627869795294),
        },
        #[cfg(test)]
        density: Polynomial {
            coefficients: [
                f64::from_bits(4598730533858807947),
                f64::from_bits(13825106300865994334),
                f64::from_bits(4596958945373350362),
                f64::from_bits(4585105489483646304),
                f64::from_bits(13812916990726693695),
                f64::from_bits(4579643246705091987),
                f64::from_bits(4575154664136128690),
                f64::from_bits(13794242323097545398),
                f64::from_bits(13773715313205182114),
                f64::from_bits(4556940123803642670),
                f64::from_bits(13766600598386723124),
                f64::from_bits(13763362377144008256),
                f64::from_bits(0),
            ],
            error: f64::from_bits(4427600618808175452),
        },
        #[cfg(test)]
        oracle_coefficients: [
            Interval {
                lo: f64::from_bits(4607696757997407216),
                hi: f64::from_bits(4607696757997407291),
            },
            Interval {
                lo: f64::from_bits(4598730533858807866),
                hi: f64::from_bits(4598730533858808028),
            },
            Interval {
                lo: f64::from_bits(13820602701238623969),
                hi: f64::from_bits(13820602701238623706),
            },
            Interval {
                lo: f64::from_bits(4589858370718374691),
                hi: f64::from_bits(4589858370718375082),
            },
            Interval {
                lo: f64::from_bits(4576098290228904339),
                hi: f64::from_bits(4576098290228906285),
            },
            Interval {
                lo: f64::from_bits(13802334511769834531),
                hi: f64::from_bits(13802334511769833434),
            },
            Interval {
                lo: f64::from_bits(4567806172475670376),
                hi: f64::from_bits(4567806172475672583),
            },
            Interval {
                lo: f64::from_bits(4562502618317481352),
                hi: f64::from_bits(4562502618317483404),
            },
            Interval {
                lo: f64::from_bits(13780731524215435676),
                hi: f64::from_bits(13780731524215432145),
            },
            Interval {
                lo: f64::from_bits(13759514375174906918),
                hi: f64::from_bits(13759514375174884147),
            },
            Interval {
                lo: f64::from_bits(4541768424889640675),
                hi: f64::from_bits(4541768424889645005),
            },
            Interval {
                lo: f64::from_bits(13750879689476474825),
                hi: f64::from_bits(13750879689476462350),
            },
            Interval {
                lo: f64::from_bits(13746969817143578214),
                hi: f64::from_bits(13746969817143569648),
            },
        ],
    },
    Patch {
        center: f64::from_bits(4610700856008900608),
        cdf: Polynomial {
            coefficients: [
                f64::from_bits(4607900637011793157),
                f64::from_bits(4596541522229231218),
                f64::from_bits(13819107082309462174),
                f64::from_bits(4590004929066028906),
                f64::from_bits(13791572435704289898),
                f64::from_bits(13800395086452980876),
                f64::from_bits(4569923564902273558),
                f64::from_bits(4555163308825143473),
                f64::from_bits(13780127212439084094),
                f64::from_bits(4542903693526526338),
                f64::from_bits(4539397810341096539),
                f64::from_bits(13754877890757550905),
                f64::from_bits(13739954791610783030),
            ],
            error: f64::from_bits(4421320424046711680),
        },
        #[cfg(test)]
        density: Polynomial {
            coefficients: [
                f64::from_bits(4596541522229231218),
                f64::from_bits(13823610681936832670),
                f64::from_bits(4597178782894831392),
                f64::from_bits(13800579634959030890),
                f64::from_bits(13810869642662009774),
                f64::from_bits(4581490078190394768),
                f64::from_bits(4567858147059637147),
                f64::from_bits(13793638011321195582),
                f64::from_bits(4557386851004192435),
                f64::from_bits(4554120989398186937),
                f64::from_bits(13770409275287947374),
                f64::from_bits(13755849308180729832),
                f64::from_bits(0),
            ],
            error: f64::from_bits(4427601462688869842),
        },
        #[cfg(test)]
        oracle_coefficients: [
            Interval {
                lo: f64::from_bits(4607900637011793113),
                hi: f64::from_bits(4607900637011793201),
            },
            Interval {
                lo: f64::from_bits(4596541522229231051),
                hi: f64::from_bits(4596541522229231386),
            },
            Interval {
                lo: f64::from_bits(13819107082309462325),
                hi: f64::from_bits(13819107082309462023),
            },
            Interval {
                lo: f64::from_bits(4590004929066028669),
                hi: f64::from_bits(4590004929066029144),
            },
            Interval {
                lo: f64::from_bits(13791572435704294883),
                hi: f64::from_bits(13791572435704284914),
            },
            Interval {
                lo: f64::from_bits(13800395086452981606),
                hi: f64::from_bits(13800395086452980145),
            },
            Interval {
                lo: f64::from_bits(4569923564902272023),
                hi: f64::from_bits(4569923564902275093),
            },
            Interval {
                lo: f64::from_bits(4555163308825137567),
                hi: f64::from_bits(4555163308825149379),
            },
            Interval {
                lo: f64::from_bits(13780127212439086727),
                hi: f64::from_bits(13780127212439081461),
            },
            Interval {
                lo: f64::from_bits(4542903693526517574),
                hi: f64::from_bits(4542903693526535103),
            },
            Interval {
                lo: f64::from_bits(4539397810341089669),
                hi: f64::from_bits(4539397810341103409),
            },
            Interval {
                lo: f64::from_bits(13754877890757556001),
                hi: f64::from_bits(13754877890757545809),
            },
            Interval {
                lo: f64::from_bits(13739954791610811788),
                hi: f64::from_bits(13739954791610754271),
            },
        ],
    },
    Patch {
        center: f64::from_bits(4611545280939032576),
        cdf: Polynomial {
            coefficients: [
                f64::from_bits(4608046770117118402),
                f64::from_bits(4594355926846065760),
                f64::from_bits(13817646902660911407),
                f64::from_bits(4589637853217213380),
                f64::from_bits(13800490294609127892),
                f64::from_bits(13797261146309202353),
                f64::from_bits(4570078212753184714),
                f64::from_bits(13775619641461127272),
                f64::from_bits(13777807741005944636),
                f64::from_bits(4546281524392130380),
                f64::from_bits(4532136999778517850),
                f64::from_bits(13754921749918087167),
                f64::from_bits(4515066245594487998),
            ],
            error: f64::from_bits(4421351735292739895),
        },
        #[cfg(test)]
        density: Polynomial {
            coefficients: [
                f64::from_bits(4594355926846065760),
                f64::from_bits(13822150502288281903),
                f64::from_bits(4596628169121608101),
                f64::from_bits(13809497493863868884),
                f64::from_bits(13808057287453534990),
                f64::from_bits(4581606064078578135),
                f64::from_bits(13788115992769548635),
                f64::from_bits(13791318539888056124),
                f64::from_bits(4560623960774575670),
                f64::from_bits(4547152642228421680),
                f64::from_bits(13770469581633684734),
                f64::from_bits(4531339889454814606),
                f64::from_bits(0),
            ],
            error: f64::from_bits(4427613630197229566),
        },
        #[cfg(test)]
        oracle_coefficients: [
            Interval {
                lo: f64::from_bits(4608046770117118352),
                hi: f64::from_bits(4608046770117118451),
            },
            Interval {
                lo: f64::from_bits(4594355926846065520),
                hi: f64::from_bits(4594355926846066001),
            },
            Interval {
                lo: f64::from_bits(13817646902660911646),
                hi: f64::from_bits(13817646902660911168),
            },
            Interval {
                lo: f64::from_bits(4589637853217212983),
                hi: f64::from_bits(4589637853217213776),
            },
            Interval {
                lo: f64::from_bits(13800490294609130088),
                hi: f64::from_bits(13800490294609125696),
            },
            Interval {
                lo: f64::from_bits(13797261146309205035),
                hi: f64::from_bits(13797261146309199671),
            },
            Interval {
                lo: f64::from_bits(4570078212753181779),
                hi: f64::from_bits(4570078212753187648),
            },
            Interval {
                lo: f64::from_bits(13775619641461150694),
                hi: f64::from_bits(13775619641461103850),
            },
            Interval {
                lo: f64::from_bits(13777807741005950035),
                hi: f64::from_bits(13777807741005939237),
            },
            Interval {
                lo: f64::from_bits(4546281524392121099),
                hi: f64::from_bits(4546281524392139661),
            },
            Interval {
                lo: f64::from_bits(4532136999778487875),
                hi: f64::from_bits(4532136999778547824),
            },
            Interval {
                lo: f64::from_bits(13754921749918098608),
                hi: f64::from_bits(13754921749918075726),
            },
            Interval {
                lo: f64::from_bits(4515066245594421631),
                hi: f64::from_bits(4515066245594554366),
            },
        ],
    },
    Patch {
        center: f64::from_bits(4612037862148276224),
        cdf: Polynomial {
            coefficients: [
                f64::from_bits(4608147904991515964),
                f64::from_bits(4591712524950108080),
                f64::from_bits(13815635194952301322),
                f64::from_bits(4588734638002040057),
                f64::from_bits(13802881301556815595),
                f64::from_bits(13791682136061376306),
                f64::from_bits(4568817836724483684),
                f64::from_bits(13781796311344194894),
                f64::from_bits(13773242825081372128),
                f64::from_bits(4546488844298298710),
                f64::from_bits(13752330915445802348),
                f64::from_bits(13752527218010124649),
                f64::from_bits(4520252744922151781),
            ],
            error: f64::from_bits(4421454995539214738),
        },
        #[cfg(test)]
        density: Polynomial {
            coefficients: [
                f64::from_bits(4591712524950108080),
                f64::from_bits(13820138794579671818),
                f64::from_bits(4595598383015219642),
                f64::from_bits(13811888500811556587),
                f64::from_bits(13802230254486189310),
                f64::from_bits(4580660782057052363),
                f64::from_bits(13794646478824075429),
                f64::from_bits(13786753623963483616),
                f64::from_bits(4560857195669015040),
                f64::from_bits(13767590994188641124),
                f64::from_bits(13768185005496053240),
                f64::from_bits(4536355663857405068),
                f64::from_bits(0),
            ],
            error: f64::from_bits(4427670390775445403),
        },
        #[cfg(test)]
        oracle_coefficients: [
            Interval {
                lo: f64::from_bits(4608147904991515896),
                hi: f64::from_bits(4608147904991516033),
            },
            Interval {
                lo: f64::from_bits(4591712524950106930),
                hi: f64::from_bits(4591712524950109229),
            },
            Interval {
                lo: f64::from_bits(13815635194952302563),
                hi: f64::from_bits(13815635194952300080),
            },
            Interval {
                lo: f64::from_bits(4588734638002037886),
                hi: f64::from_bits(4588734638002042228),
            },
            Interval {
                lo: f64::from_bits(13802881301556821935),
                hi: f64::from_bits(13802881301556809255),
            },
            Interval {
                lo: f64::from_bits(13791682136061392454),
                hi: f64::from_bits(13791682136061360157),
            },
            Interval {
                lo: f64::from_bits(4568817836724474496),
                hi: f64::from_bits(4568817836724492872),
            },
            Interval {
                lo: f64::from_bits(13781796311344213907),
                hi: f64::from_bits(13781796311344175882),
            },
            Interval {
                lo: f64::from_bits(13773242825081408378),
                hi: f64::from_bits(13773242825081335879),
            },
            Interval {
                lo: f64::from_bits(4546488844298266550),
                hi: f64::from_bits(4546488844298330869),
            },
            Interval {
                lo: f64::from_bits(13752330915446016406),
                hi: f64::from_bits(13752330915445588289),
            },
            Interval {
                lo: f64::from_bits(13752527218010208711),
                hi: f64::from_bits(13752527218010040587),
            },
            Interval {
                lo: f64::from_bits(4520252744922026492),
                hi: f64::from_bits(4520252744922277070),
            },
        ],
    },
    Patch {
        center: f64::from_bits(4612460074613342208),
        cdf: Polynomial {
            coefficients: [
                f64::from_bits(4608215486743635650),
                f64::from_bits(4589286680303653554),
                f64::from_bits(13813453168034138344),
                f64::from_bits(4587083678003856986),
                f64::from_bits(13803528823966462382),
                f64::from_bits(4548142929080243814),
                f64::from_bits(4566844173877811450),
                f64::from_bits(13782929595339527234),
                f64::from_bits(13757998485613258167),
                f64::from_bits(4544897442812139459),
                f64::from_bits(13758554796838484190),
                f64::from_bits(13746502367848712202),
                f64::from_bits(4520487485065108786),
            ],
            error: f64::from_bits(4421524168050972389),
        },
        #[cfg(test)]
        density: Polynomial {
            coefficients: [
                f64::from_bits(4589286680303653554),
                f64::from_bits(13817956767661508840),
                f64::from_bits(4594360163016582340),
                f64::from_bits(13812536023221203374),
                f64::from_bits(4558460788703311360),
                f64::from_bits(4578200248808301942),
                f64::from_bits(13795638102319991226),
                f64::from_bits(13771509284495369655),
                f64::from_bits(4559066868997085884),
                f64::from_bits(13773205979432843030),
                f64::from_bits(13762270631008822030),
                f64::from_bits(4536531718964622821),
                f64::from_bits(0),
            ],
            error: f64::from_bits(4427718888006158298),
        },
        #[cfg(test)]
        oracle_coefficients: [
            Interval {
                lo: f64::from_bits(4608215486743635569),
                hi: f64::from_bits(4608215486743635731),
            },
            Interval {
                lo: f64::from_bits(4589286680303651857),
                hi: f64::from_bits(4589286680303655251),
            },
            Interval {
                lo: f64::from_bits(13813453168034140335),
                hi: f64::from_bits(13813453168034136354),
            },
            Interval {
                lo: f64::from_bits(4587083678003853307),
                hi: f64::from_bits(4587083678003860666),
            },
            Interval {
                lo: f64::from_bits(13803528823966473663),
                hi: f64::from_bits(13803528823966451101),
            },
            Interval {
                lo: f64::from_bits(4548142929079284425),
                hi: f64::from_bits(4548142929081203204),
            },
            Interval {
                lo: f64::from_bits(4566844173877793720),
                hi: f64::from_bits(4566844173877829179),
            },
            Interval {
                lo: f64::from_bits(13782929595339565258),
                hi: f64::from_bits(13782929595339489210),
            },
            Interval {
                lo: f64::from_bits(13757998485614457384),
                hi: f64::from_bits(13757998485612058950),
            },
            Interval {
                lo: f64::from_bits(4544897442812070846),
                hi: f64::from_bits(4544897442812208072),
            },
            Interval {
                lo: f64::from_bits(13758554796838601816),
                hi: f64::from_bits(13758554796838366565),
            },
            Interval {
                lo: f64::from_bits(13746502367849092342),
                hi: f64::from_bits(13746502367848332063),
            },
            Interval {
                lo: f64::from_bits(4520487485064817715),
                hi: f64::from_bits(4520487485065399856),
            },
        ],
    },
    Patch {
        center: f64::from_bits(4612882287078408192),
        cdf: Polynomial {
            coefficients: [
                f64::from_bits(4608259091521007146),
                f64::from_bits(4586014085980452750),
                f64::from_bits(13810940896336153424),
                f64::from_bits(4585435809316812190),
                f64::from_bits(13802939280425099994),
                f64::from_bits(4566398047396603234),
                f64::from_bits(4562923742472546561),
                f64::from_bits(13782415073865034236),
                f64::from_bits(4545519447932189178),
                f64::from_bits(4540994834279484025),
                f64::from_bits(13759059540456201717),
                f64::from_bits(4514671954961624254),
                f64::from_bits(4518054974693817820),
            ],
            error: f64::from_bits(4421613790318396098),
        },
        #[cfg(test)]
        density: Polynomial {
            coefficients: [
                f64::from_bits(4586014085980452750),
                f64::from_bits(13815444495963523920),
                f64::from_bits(4592576903084691565),
                f64::from_bits(13811946479679840986),
                f64::from_bits(4576625637092091741),
                f64::from_bits(4574571401514089857),
                f64::from_bits(13795187896029809852),
                f64::from_bits(4559030246814300666),
                f64::from_bits(4555239384351269832),
                f64::from_bits(13773836908954989938),
                f64::from_bits(4530633625982431939),
                f64::from_bits(4534289847610219722),
                f64::from_bits(0),
            ],
            error: f64::from_bits(4427843247659810490),
        },
        #[cfg(test)]
        oracle_coefficients: [
            Interval {
                lo: f64::from_bits(4608259091521007055),
                hi: f64::from_bits(4608259091521007238),
            },
            Interval {
                lo: f64::from_bits(4586014085980446568),
                hi: f64::from_bits(4586014085980458931),
            },
            Interval {
                lo: f64::from_bits(13810940896336161249),
                hi: f64::from_bits(13810940896336145598),
            },
            Interval {
                lo: f64::from_bits(4585435809316804554),
                hi: f64::from_bits(4585435809316819826),
            },
            Interval {
                lo: f64::from_bits(13802939280425124542),
                hi: f64::from_bits(13802939280425075445),
            },
            Interval {
                lo: f64::from_bits(4566398047396467157),
                hi: f64::from_bits(4566398047396739310),
            },
            Interval {
                lo: f64::from_bits(4562923742472462967),
                hi: f64::from_bits(4562923742472630155),
            },
            Interval {
                lo: f64::from_bits(13782415073865127094),
                hi: f64::from_bits(13782415073864941377),
            },
            Interval {
                lo: f64::from_bits(4545519447931810825),
                hi: f64::from_bits(4545519447932567530),
            },
            Interval {
                lo: f64::from_bits(4540994834279126755),
                hi: f64::from_bits(4540994834279841295),
            },
            Interval {
                lo: f64::from_bits(13759059540456517112),
                hi: f64::from_bits(13759059540455886322),
            },
            Interval {
                lo: f64::from_bits(4514671954957431008),
                hi: f64::from_bits(4514671954965817501),
            },
            Interval {
                lo: f64::from_bits(4518054974692993265),
                hi: f64::from_bits(4518054974694642374),
            },
        ],
    },
    Patch {
        center: f64::from_bits(4613304499543474176),
        cdf: Polynomial {
            coefficients: [
                f64::from_bits(4608286256836354398),
                f64::from_bits(4582813279591732700),
                f64::from_bits(13808396741187037916),
                f64::from_bits(4583280333519386804),
                f64::from_bits(13801645737797357614),
                f64::from_bits(4568484860169481202),
                f64::from_bits(4556037791610235256),
                f64::from_bits(13780919216943564496),
                f64::from_bits(4548570213731592426),
                f64::from_bits(4532233143535912414),
                f64::from_bits(13757510924580545412),
                f64::from_bits(4523039095500107448),
                f64::from_bits(4512299398525271713),
            ],
            error: f64::from_bits(4421823154370964746),
        },
        #[cfg(test)]
        density: Polynomial {
            coefficients: [
                f64::from_bits(4582813279591732700),
                f64::from_bits(13812900340814408412),
                f64::from_bits(4590381754746387079),
                f64::from_bits(13810652937052098606),
                f64::from_bits(4579076668835014382),
                f64::from_bits(4567698048500838170),
                f64::from_bits(13793316071270102518),
                f64::from_bits(4562081012613703914),
                f64::from_bits(4546508382171594394),
                f64::from_bits(13772517349758119474),
                f64::from_bits(4538773145350031421),
                f64::from_bits(4528138854246059769),
                f64::from_bits(0),
            ],
            error: f64::from_bits(4428012989907866769),
        },
        #[cfg(test)]
        oracle_coefficients: [
            Interval {
                lo: f64::from_bits(4608286256836354272),
                hi: f64::from_bits(4608286256836354525),
            },
            Interval {
                lo: f64::from_bits(4582813279591712923),
                hi: f64::from_bits(4582813279591752478),
            },
            Interval {
                lo: f64::from_bits(13808396741187051360),
                hi: f64::from_bits(13808396741187024471),
            },
            Interval {
                lo: f64::from_bits(4583280333519359136),
                hi: f64::from_bits(4583280333519414472),
            },
            Interval {
                lo: f64::from_bits(13801645737797404190),
                hi: f64::from_bits(13801645737797311037),
            },
            Interval {
                lo: f64::from_bits(4568484860169346693),
                hi: f64::from_bits(4568484860169615710),
            },
            Interval {
                lo: f64::from_bits(4556037791609548938),
                hi: f64::from_bits(4556037791610921573),
            },
            Interval {
                lo: f64::from_bits(13780919216943959162),
                hi: f64::from_bits(13780919216943169830),
            },
            Interval {
                lo: f64::from_bits(4548570213730761787),
                hi: f64::from_bits(4548570213732423066),
            },
            Interval {
                lo: f64::from_bits(4532233143532677185),
                hi: f64::from_bits(4532233143539147644),
            },
            Interval {
                lo: f64::from_bits(13757510924582015670),
                hi: f64::from_bits(13757510924579075153),
            },
            Interval {
                lo: f64::from_bits(4523039095497595091),
                hi: f64::from_bits(4523039095502619805),
            },
            Interval {
                lo: f64::from_bits(4512299398517153802),
                hi: f64::from_bits(4512299398533389624),
            },
        ],
    },
    Patch {
        center: f64::from_bits(4613726712008540160),
        cdf: Polynomial {
            coefficients: [
                f64::from_bits(4608302597564548984),
                f64::from_bits(4579600211082108176),
                f64::from_bits(13805166233302247326),
                f64::from_bits(4580898533535746720),
                f64::from_bits(13800096281974495974),
                f64::from_bits(4568808547235925246),
                f64::from_bits(13771191422596655352),
                f64::from_bits(13777502363266567431),
                f64::from_bits(4548490517350660104),
                f64::from_bits(13755288771679387022),
                f64::from_bits(13754106228989203792),
                f64::from_bits(4523887188123693404),
                f64::from_bits(13725717662784615114),
            ],
            error: f64::from_bits(4422705506392029377),
        },
        #[cfg(test)]
        density: Polynomial {
            coefficients: [
                f64::from_bits(4579600211082108176),
                f64::from_bits(13809669832929617822),
                f64::from_bits(4588022789226778608),
                f64::from_bits(13809103481229236966),
                f64::from_bits(4579481277668069438),
                f64::from_bits(13782654481140661946),
                f64::from_bits(13790326324302730086),
                f64::from_bits(4562001316232771592),
                f64::from_bits(13769524459226156352),
                f64::from_bits(13768771169528085156),
                f64::from_bits(4539783848548461038),
                f64::from_bits(13741793761840576024),
                f64::from_bits(0),
            ],
            error: f64::from_bits(4428868667677060449),
        },
        #[cfg(test)]
        oracle_coefficients: [
            Interval {
                lo: f64::from_bits(4608302597564548681),
                hi: f64::from_bits(4608302597564549287),
            },
            Interval {
                lo: f64::from_bits(4579600211081993778),
                hi: f64::from_bits(4579600211082222575),
            },
            Interval {
                lo: f64::from_bits(13805166233302330446),
                hi: f64::from_bits(13805166233302164206),
            },
            Interval {
                lo: f64::from_bits(4580898533535656662),
                hi: f64::from_bits(4580898533535836778),
            },
            Interval {
                lo: f64::from_bits(13800096281974654548),
                hi: f64::from_bits(13800096281974337400),
            },
            Interval {
                lo: f64::from_bits(4568808547235448491),
                hi: f64::from_bits(4568808547236402001),
            },
            Interval {
                lo: f64::from_bits(13771191422606751372),
                hi: f64::from_bits(13771191422586559331),
            },
            Interval {
                lo: f64::from_bits(13777502363268069398),
                hi: f64::from_bits(13777502363265065464),
            },
            Interval {
                lo: f64::from_bits(4548490517347395839),
                hi: f64::from_bits(4548490517353924369),
            },
            Interval {
                lo: f64::from_bits(13755288771692492493),
                hi: f64::from_bits(13755288771666281551),
            },
            Interval {
                lo: f64::from_bits(13754106228995333827),
                hi: f64::from_bits(13754106228983073757),
            },
            Interval {
                lo: f64::from_bits(4523887188112926005),
                hi: f64::from_bits(4523887188134460802),
            },
            Interval {
                lo: f64::from_bits(13725717662927505209),
                hi: f64::from_bits(13725717662641725018),
            },
        ],
    },
];
