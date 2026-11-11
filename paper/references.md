# References the project builds on (IEEE style)

## Datasets
1. I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, "Toward generating a new intrusion detection dataset and intrusion traffic characterization," in *Proc. 4th Int. Conf. Information Systems Security and Privacy (ICISSP)*, 2018, pp. 108–116. **CIC-IDS2017, the main dataset.**
2. C. Torrano Giménez, A. Pérez Villegas, and G. Álvarez Marañón, "HTTP data set CSIC 2010," Spanish National Research Council (CSIC), 2010. **Request-level detection.**
3. Morzeux, "HttpParamsDataset," GitHub, 2016 (commit 926670a). **Payloads: SQLi, XSS, command injection, path traversal.**
3a. T. Sureda Riera, J. R. Bermejo Higuera, J. Bermejo Higuera, J. A. Sicilia Montalvo, and J. J. Martínez Herráiz, "A new multi-label dataset for web attacks CAPEC classification using machine learning techniques," *Computers & Security*, vol. 120, art. 102788, 2022, doi: 10.1016/j.cose.2022.102788. Data: SR-BH 2020, Harvard Dataverse, doi: 10.7910/DVN/OGOIXX (CC0). **Second request-level dataset, combined with CSIC 2010.**

## Problems in the dataset (the validity audit, E7)
4. G. Engelen, V. Rimmer, and W. Joosen, "Troubleshooting an intrusion detection dataset: the CICIDS2017 case study," in *Proc. IEEE Security and Privacy Workshops (SPW)*, 2021, pp. 7–12, doi: 10.1109/SPW53761.2021.00009.
5. M. Lanvin, P.-F. Gimenez, Y. Han, F. Majorczyk, L. Mé, and É. Totel, "Errors in the CICIDS2017 dataset and the significant differences in detection performances it makes," in *Risks and Security of Internet and Systems (CRiSIS 2022)*, LNCS vol. 13857, Springer, 2023, doi: 10.1007/978-3-031-31108-6_2.

## Machine learning on CIC-IDS2017
6. Z. K. Maseer, R. Yusof, N. Bahaman, S. A. Mostafa, and C. F. M. Foozy, "Benchmarking of machine learning for anomaly based intrusion detection systems in the CICIDS2017 dataset," *IEEE Access*, vol. 9, pp. 22351–22370, 2021.
7. Kurniabudi, D. Stiawan, Darmawijoyo, M. Y. Bin Idris, A. M. Bamhdi, and R. Budiarto, "CICIDS-2017 dataset feature analysis with information gain for anomaly detection," *IEEE Access*, vol. 8, pp. 132911–132921, 2020, doi: 10.1109/ACCESS.2020.3009843.

## Machine learning for web attacks (SQLi, XSS, WAF)
8. J. R. Tadhani, V. Vekariya, V. Sorathiya, S. Alshathri, and W. El-Shafai, "Securing web applications against XSS and SQLi attacks using a novel deep learning approach," *Scientific Reports* (Springer Nature), vol. 14, 2024, doi: 10.1038/s41598-023-48845-4.
9. J. Yang, Y. Wu, Y. Yuan, H. Xue, S. Bourouis, M. Abdel-Salam, S. Prajapat, and L. Y. Por, "LLM-AE-MP: Web attack detection using a large language model with autoencoder and multilayer perceptron," *Expert Systems with Applications* (Elsevier), vol. 274, art. 126982, 2025.
10. Z. Qu, X. Ling, T. Wang, X. Chen, S. Ji, and C. Wu, "AdvSQLi: Generating adversarial SQL injections against real-world WAF-as-a-service," *IEEE Trans. Information Forensics and Security*, vol. 19, pp. 2623–2638, 2024, doi: 10.1109/TIFS.2024.3350911.
11. G. Floris, C. Scano, B. Montaruli, L. Demetrio, A. Valenza, L. Compagna, D. Ariu, L. Piras, D. Balzarotti, and B. Biggio, "ModSec-AdvLearn: Countering adversarial SQL injections with robust machine learning," *IEEE Trans. Information Forensics and Security*, 2025.
12. J. Liang, W. Zhao, and W. Ye, "Anomaly-based web attack detection: A deep learning approach," in *Proc. 6th Int. Conf. Network, Communication and Computing (ICNCC)*, ACM, 2017.
13. F. A. Mereani and J. M. Howe, "Detecting cross-site scripting attacks using machine learning," in *Advances in Intelligent Systems and Computing*, vol. 723, Springer, 2018, pp. 200–210, doi: 10.1007/978-3-319-74690-6_20.
14. C. Kruegel and G. Vigna, "Anomaly detection of web-based attacks," in *Proc. 10th ACM Conf. Computer and Communications Security (CCS)*, 2003, pp. 251–261, doi: 10.1145/948109.948144. **The source of the learned per-parameter profile.**

## AI-assisted penetration testing
15. G. Deng et al., "PentestGPT: Evaluating and harnessing large language models for automated penetration testing," in *Proc. 33rd USENIX Security Symposium*, 2024, pp. 847–864.
16. X. Shen, L. Wang, Z. Li, Y. Chen, W. Zhao, D. Sun, J. Wang, and W. Ruan, "PentestAgent: Incorporating LLM agents to automated penetration testing," in *Proc. ACM Asia Conf. Computer and Communications Security (ASIA CCS)*, 2025, doi: 10.1145/3708821.3733882.
17. Y. Guo, L. Yu, S. Li, M. Zhou, and G. Chu, "Decoupling reconnaissance and exploitation: Measuring the capability boundaries of LLM-based web penetration testing," in *Proc. ICIC 2026*, LNCS vol. 16652, Springer, 2026, pp. 607–618.

## Method and pitfalls
18. D. Arp et al., "Dos and don'ts of machine learning in computer security," in *Proc. 31st USENIX Security Symposium*, 2022.
19. R. Sommer and V. Paxson, "Outside the closed world: On using machine learning for network intrusion detection," in *Proc. IEEE Symposium on Security and Privacy*, 2010, pp. 305–316.
20. D. E. Denning, "An intrusion-detection model," *IEEE Trans. Software Engineering*, vol. SE-13, no. 2, pp. 222–232, 1987. **The conventional threshold IDS.**

## Tools and libraries
21. T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD*, 2016, pp. 785–794, doi: 10.1145/2939672.2939785.
22. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *J. Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
23. OWASP Foundation, "OWASP ModSecurity Core Rule Set," https://coreruleset.org. **The signature baseline is modelled on its rule families.**
24. OWASP Foundation, "ZAP (Zed Attack Proxy) 2.17," https://www.zaproxy.org. **The conventional scanner.**
25. OWASP Foundation, "OWASP Juice Shop 20.3," https://owasp.org/www-project-juice-shop. **The intentionally vulnerable target.**
26. OWASP Foundation, "OWASP Top 10: 2021," https://owasp.org/Top10.
