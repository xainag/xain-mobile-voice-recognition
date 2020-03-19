import 'package:flutter/material.dart';

class WordTile extends StatelessWidget {
  const WordTile(
    this.data, {
    Key key,
    this.isSelected = false,
  }) : super(key: key);

  final String data;
  final bool isSelected;

  @override
  Widget build(BuildContext context) {
    return AnimatedContainer(
      duration: Duration(milliseconds: 100),
      child: Text(
        data,
        style: TextStyle(
          color: isSelected ? Colors.blueGrey[100] : Colors.black87,
          fontSize: 12,
        ),
      ),
      padding: EdgeInsets.symmetric(
        horizontal: 16.0,
        vertical: 10.0,
      ),
      decoration: BoxDecoration(
        color: isSelected ? Colors.blueGrey[700] : Colors.blueGrey[200],
        boxShadow: [
          BoxShadow(
            offset: Offset(2, 2.5),
            color: Colors.black12,
            blurRadius: 1.5,
          ),
        ],
        borderRadius: BorderRadius.all(
          Radius.circular(2),
        ),
      ),
    );
  }
}
