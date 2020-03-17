import 'package:flutter/material.dart';

class InfoText extends StatelessWidget {
  const InfoText(
    this.data, {
    Key key,
  }) : super(key: key);

  final String data;

  @override
  Widget build(BuildContext context) {
    return Text(
      data,
      softWrap: true,
      strutStyle: StrutStyle(height: 1.25),
      textAlign: TextAlign.center,
      style: TextStyle(
        fontSize: 12,
        fontWeight: FontWeight.w600,
        color: Colors.black54,
      ),
    );
  }
}
